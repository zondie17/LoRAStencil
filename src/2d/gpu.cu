#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include "2d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_ROW 32
#define BLOCK_SIZE_COL 64
#define HALO 4
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2 + 4)
#define TENSOR_CORE_M 8
#define TENSOR_CORE_N 4
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 8
#define CALC_NUM 4
#define DATA_NUM 5

__constant__ double param_matrix_u_d[4 * 16 * 8];
__constant__ double param_matrix_v_d[4 * 8 * 16];


__constant__ double param_matrix_star2d3r_horizontal_d[16 * 8];
__constant__ double param_matrix_star2d3r_vertical_d[16 * 8];

__constant__ double param_matrix_star2d1r_horizontal_d[2 * 8 * 16];
__constant__ double param_matrix_star2d1r_vertical_d[2 * 16 * 8];



__global__ void
breakdown1(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        sharedmem[i] = in[begin + IDX(row, col, ldm)];
    }
    __syncthreads();

    int row = tid / 8;
    int col = (tid % 8) * 8;
    for (int i = 0; i < 8; ++i) {
        double res = 0;
        for (int t = 0; t < 3; t++) {
            // vertical gather
            double temp[7] = {};
            for (int x = 0; x < 7; x++) {
                for (int y = 0; y < 7; y++) {
                    temp[x] += sharedmem[IDX(HALO + row + x - 3, HALO + col + y - 3, D_BLOCK_SIZE_ROW)] *
                               param_matrix_v_d[t * 128 + y + 1];
                }
            }

            // horizontal gather
            for (int x = 0; x < 7; x++) {
                res += temp[x] * param_matrix_u_d[t * 128 + (x + 1) * 8];
            }
        }
        out[begin + IDX(HALO + row, HALO + col + i, ldm)] = res;
        col++;
    }

    __syncthreads();
}

__global__ void
breakdown2(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();

    int row = tid / 8;
    int col = (tid % 8) * 8;
    for (int i = 0; i < 8; ++i) {
        double res = 0;
        for (int t = 0; t < 3; t++) {
            // vertical gather
            double temp[7] = {};
            for (int x = 0; x < 7; x++) {
                for (int y = 0; y < 7; y++) {
                    temp[x] += sharedmem[IDX(HALO + row + x - 3, HALO + col + y - 3, D_BLOCK_SIZE_ROW)] *
                               param_matrix_v_d[t * 128 + y + 1];
                }
            }

            // horizontal gather
            for (int x = 0; x < 7; x++) {
                res += temp[x] * param_matrix_u_d[t * 128 + (x + 1) * 8];
            }
        }
        out[begin + IDX(HALO + row, HALO + col + i, ldm)] = res;
    }

    __syncthreads();
}

__global__ void
breakdown3(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int warp_begin = IDX((warp_id / 2) * 8, (warp_id % 2) * 32, D_BLOCK_SIZE_COL);
    int calc_times = 0;
    int calc_begin = 8 * calc_times;
    int lane = threadIdx.x % 32;

    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag[2][4];
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in2_frag[2];
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1[4][4];
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2[3][4];
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_v_frag[2];
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> acc_temp_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> temp_frag[2][2];


    // the first time
    int t = 0;
    wmma::fill_fragment(acc_v_frag[0], 0.0);
    wmma::fill_fragment(acc_v_frag[1], 0.0);
    wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::load_matrix_sync(param_frag1[t][i], param_matrix_v_d + t * 128 + i * 4, 16);
        wmma::load_matrix_sync(in_frag[(calc_times) % 2][i],
                               sharedmem + warp_begin + calc_begin + IDX(4 * i, 0, D_BLOCK_SIZE_COL),
                               D_BLOCK_SIZE_COL);
        wmma::mma_sync(acc_v_frag[0], param_frag1[t][i], in_frag[(calc_times) % 2][i], acc_v_frag[0]);
        wmma::load_matrix_sync(in_frag[(calc_times + 1) % 2][i],
                               sharedmem + warp_begin + calc_begin + IDX(4 * i, 8, D_BLOCK_SIZE_COL),
                               D_BLOCK_SIZE_COL);
        wmma::mma_sync(acc_v_frag[1], param_frag1[t][i], in_frag[(calc_times + 1) % 2][i], acc_v_frag[1]);
    }




    int row = lane % 8, col = lane / 4;
    temp_frag[0][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane >= 16], row * 4 + col / 2, 32);
    temp_frag[1][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane >= 16], row * 4 + col / 2, 32);
    row += lane % 8 < 4 ? 4 : -4;
    temp_frag[0][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane < 16], row * 4 + col / 2, 32);
    temp_frag[1][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane < 16], row * 4 + col / 2, 32);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::load_matrix_sync(param_frag2[t][i], param_matrix_u_d + t * 128 + i * 32, 8);
        wmma::mma_sync(acc_frag, temp_frag[i / 2][i % 2], param_frag2[t][i], acc_frag);
    }


#pragma unroll
    for (t = 1; t < 3; t++) {
        wmma::fill_fragment(acc_v_frag[0], 0.0);
        wmma::fill_fragment(acc_v_frag[1], 0.0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(param_frag1[t][i], param_matrix_v_d + t * 128 + i * 4, 16);
            wmma::mma_sync(acc_v_frag[0], param_frag1[t][i], in_frag[(calc_times) % 2][i], acc_v_frag[0]);
            wmma::mma_sync(acc_v_frag[1], param_frag1[t][i], in_frag[(calc_times + 1) % 2][i], acc_v_frag[1]);
        }



        temp_frag[0][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane >= 16], row * 4 + col / 2, 32);
        temp_frag[1][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane >= 16], row * 4 + col / 2, 32);
        row += lane % 8 < 4 ? 4 : -4;
        temp_frag[0][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane < 16], row * 4 + col / 2, 32);
        temp_frag[1][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane < 16], row * 4 + col / 2, 32);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(param_frag2[t][i], param_matrix_u_d + t * 128 + i * 32, 8);
            wmma::mma_sync(acc_frag, temp_frag[i / 2][i % 2], param_frag2[t][i], acc_frag);
        }

    }
    t = 3;
#pragma unroll
    for (int i = 1; i < 3; i++) {
        wmma::load_matrix_sync(param_frag1[t][i], param_matrix_v_d + t * 128 + i * 4, 16);
        wmma::load_matrix_sync(in2_frag[i],
                               sharedmem + warp_begin + calc_begin + IDX(4 * (i + 1), 4, D_BLOCK_SIZE_COL),
                               D_BLOCK_SIZE_COL);
        wmma::mma_sync(acc_frag, param_frag1[t][i], in2_frag[i], acc_frag);
    }

    wmma::store_matrix_sync(
            out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(calc_times, 4 * ldm + 4, 8),
            acc_frag, ldm, wmma::mem_row_major);

    // the last three times
#pragma unroll
    for (calc_times = 1; calc_times < 4; calc_times++) {
        t = 0;
        wmma::fill_fragment(acc_v_frag[0], 0.0);
        wmma::fill_fragment(acc_v_frag[1], 0.0);
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(acc_v_frag[0], param_frag1[t][i], in_frag[(calc_times) % 2][i], acc_v_frag[0]);
            wmma::load_matrix_sync(in_frag[(calc_times + 1) % 2][i],
                                   sharedmem + warp_begin + calc_begin + IDX(4 * i, 8, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
            wmma::mma_sync(acc_v_frag[1], param_frag1[t][i], in_frag[(calc_times + 1) % 2][i], acc_v_frag[1]);
        }


        temp_frag[0][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane >= 16], row * 4 + col / 2, 32);
        temp_frag[1][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane >= 16], row * 4 + col / 2, 32);
        row += lane % 8 < 4 ? 4 : -4;
        temp_frag[0][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane < 16], row * 4 + col / 2, 32);
        temp_frag[1][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane < 16], row * 4 + col / 2, 32);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::mma_sync(acc_frag, temp_frag[i / 2][i % 2], param_frag2[t][i], acc_frag);
        }

#pragma unroll
        for (t = 1; t < 3; t++) {
            wmma::fill_fragment(acc_v_frag[0], 0.0);
            wmma::fill_fragment(acc_v_frag[1], 0.0);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                wmma::mma_sync(acc_v_frag[0], param_frag1[t][i], in_frag[(calc_times) % 2][i], acc_v_frag[0]);
                wmma::mma_sync(acc_v_frag[1], param_frag1[t][i], in_frag[(calc_times + 1) % 2][i], acc_v_frag[1]);
            }


            temp_frag[0][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane >= 16], row * 4 + col / 2, 32);
            temp_frag[1][lane % 8 >= 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane >= 16], row * 4 + col / 2, 32);
            row += lane % 8 < 4 ? 4 : -4;
            temp_frag[0][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[0].x[lane < 16], row * 4 + col / 2, 32);
            temp_frag[1][lane % 8 < 4].x[0] = __shfl_sync(0xffffffff, acc_v_frag[1].x[lane < 16], row * 4 + col / 2, 32);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                wmma::mma_sync(acc_frag, temp_frag[i / 2][i % 2], param_frag2[t][i], acc_frag);
            }

        }
        t = 3;
#pragma unroll
        for (int i = 1; i < 3; i++) {
            wmma::load_matrix_sync(in2_frag[i],
                                   sharedmem + warp_begin + calc_begin + IDX(4 * (i + 1), 4, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
            wmma::mma_sync(acc_frag, param_frag1[t][i], in2_frag[i], acc_frag);
        }
        wmma::store_matrix_sync(
                out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(calc_times, 4 * ldm + 4, 8),
                acc_frag, ldm, wmma::mem_row_major);
    }


}







__global__ void
kernel2d_star2d1r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int warp_begin = IDX((warp_id / 2) * 8, (warp_id % 2) * BLOCK_SIZE_COL / 2, D_BLOCK_SIZE_COL);


    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag[5][4];
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1[2][4];
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2[2][4];
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_v_frag[2];
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> acc_temp_frag;

#pragma unroll
    for (int t = 0; t < 5; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(in_frag[t][i],
                                   sharedmem + warp_begin + 8 * t + IDX(4 * i, 0, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
        }
    }
#pragma unroll
    for (int t = 0; t < 2; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(param_frag1[t][i], param_matrix_star2d1r_horizontal_d + t * 128 + i * 4, 16);
            wmma::load_matrix_sync(param_frag2[t][i], param_matrix_star2d1r_vertical_d + t * 128 + i * 32, 8);
        }
    }

#pragma unroll
    for (int calc = 0; calc < 4; calc++) {
        wmma::fill_fragment(acc_frag, 0.0);
        int data_begin = warp_begin + IDX(calc, 4 * D_BLOCK_SIZE_COL + 4, 8);
        int row = (tid % 32) / 4;
        int col = ((tid % 32) % 4) * 2;
        for (int num = 0; num < 2; num++) {
            acc_frag.x[num] =
                    acc_frag.x[num] + sharedmem[data_begin + IDX(row, col + num, D_BLOCK_SIZE_COL) - 3] +
                    sharedmem[data_begin + IDX(row, col + num, D_BLOCK_SIZE_COL) + 3] +
                    sharedmem[data_begin + IDX(row, col + num, D_BLOCK_SIZE_COL) - 3 * D_BLOCK_SIZE_COL] +
                    sharedmem[data_begin + IDX(row, col + num, D_BLOCK_SIZE_COL) + 3 * D_BLOCK_SIZE_COL];
        }
#pragma unroll
        for (int t = 0; t < 2; t++) {
            wmma::fill_fragment(acc_v_frag[0], 0.0);
            wmma::fill_fragment(acc_v_frag[1], 0.0);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                wmma::mma_sync(acc_v_frag[0], param_frag1[t][i], in_frag[calc][i], acc_v_frag[0]);
                wmma::mma_sync(acc_v_frag[1], param_frag1[t][i], in_frag[calc + 1][i], acc_v_frag[1]);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_temp_frag.x[0] = acc_v_frag[i / 2].x[i % 2];
                wmma::mma_sync(acc_frag, acc_temp_frag, param_frag2[t][i], acc_frag);
            }
        }

        wmma::store_matrix_sync(
                out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(calc, 4 * ldm + 4, 8),
                acc_frag, ldm, wmma::mem_row_major);
    }

}

__global__ void
kernel2d_star2d3r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
//        int row = (i * 862) >> 16;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int warp_begin = IDX((warp_id / 2) * 8, (warp_id % 2) * BLOCK_SIZE_COL / 2, D_BLOCK_SIZE_COL);

    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag1;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag2;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag[4];

    // part1
#pragma unroll
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag1, param_matrix_star2d3r_horizontal_d + row * 4, 16);
#pragma unroll
        for (int data = 0; data < 4; data++) {
            wmma::load_matrix_sync(in_frag1,
                                   sharedmem + warp_begin + 8 * data +
                                   IDX(4 * row, 4, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
            wmma::mma_sync(acc_frag[data], param_frag1, in_frag1, acc_frag[data]);
        }
    }

    // part2
#pragma unroll
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag2, param_matrix_star2d3r_vertical_d + row * 32, 8);
#pragma unroll
        for (int data = 0; data < 4; data++) {
            wmma::load_matrix_sync(in_frag2,
                                   sharedmem + warp_begin + 8 * data +
                                   IDX(4, 4 * row, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
            wmma::mma_sync(acc_frag[data], in_frag2, param_frag2, acc_frag[data]);
        }
    }

#pragma unroll
    for (int data = 0; data < 4; data++) {
        wmma::store_matrix_sync(
                out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(data, 4 * ldm + 4, 8),
                acc_frag[data], ldm, wmma::mem_row_major);
    }
}

__global__ void
kernel2d_box2d3r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
//        int row = (i * 862) >> 16;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int warp_begin = IDX((warp_id / 2) * 8, (warp_id % 2) * BLOCK_SIZE_COL / 2, D_BLOCK_SIZE_COL);

    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_v_frag[5];
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag[4];
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> acc_temp_frag;

//    wmma::fill_fragment(acc_frag[0], 0.0);
//    wmma::fill_fragment(acc_frag[1], 0.0);
//    wmma::fill_fragment(acc_frag[2], 0.0);
//    wmma::fill_fragment(acc_frag[3], 0.0);

#pragma unroll
    for (int param = 0; param < 3; param++) {

        wmma::fill_fragment(acc_v_frag[0], 0.0);
        wmma::fill_fragment(acc_v_frag[1], 0.0);
        wmma::fill_fragment(acc_v_frag[2], 0.0);
        wmma::fill_fragment(acc_v_frag[3], 0.0);
        wmma::fill_fragment(acc_v_frag[4], 0.0);


        // part1
#pragma unroll
        for (int row = 0; row < 4; row++) {
            wmma::load_matrix_sync(param_frag1, param_matrix_v_d + param * 128 + row * 4, 16);
#pragma unroll
            for (int data = 0; data < 5; data++) {
                wmma::load_matrix_sync(in_frag,
                                       sharedmem + warp_begin + 8 * data +
                                       IDX(4 * row, 0, D_BLOCK_SIZE_COL),
                                       D_BLOCK_SIZE_COL);
                wmma::mma_sync(acc_v_frag[data], param_frag1, in_frag, acc_v_frag[data]);
            }
        }

        // part2
#pragma unroll
        for (int row = 0; row < 4; row++) {
            wmma::load_matrix_sync(param_frag2, param_matrix_u_d + param * 128 + row * 32, 8);
#pragma unroll
            for (int data = 0; data < 4; data++) {
                acc_temp_frag.x[0] = acc_v_frag[data + row / 2].x[row % 2];
                wmma::mma_sync(acc_frag[data], acc_temp_frag, param_frag2, acc_frag[data]);
            }
        }

    }

#pragma unroll
    for (int data = 0; data < 4; data++) {
        wmma::store_matrix_sync(
                out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(data, 4 * ldm + 4, 8),
                acc_frag[data], ldm, wmma::mem_row_major);
    }

}







void gpu_star_2d1r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                   const int times, const int input_m, const int input_n) {
    double param_matrix_horizontal_h[2][8 * 16] = {0.0};
    double param_matrix_vertical_h[2][16 * 8] = {0.0};
    double param_matrix_vertical2_h[2][16 * 8] = {0.0};

    double horizontal_h[2][7] = {0.0};
    double vertical_h[2][7] = {0.0};

    horizontal_h[0][1] = 2.0;
    horizontal_h[0][2] = 4.0;
    horizontal_h[0][3] = 5.0;
    horizontal_h[0][4] = 4.0;
    horizontal_h[0][5] = 2.0;
    horizontal_h[1][2] = 2.0;
    horizontal_h[1][3] = 3.0;
    horizontal_h[1][4] = 2.0;
    vertical_h[0][2] = 1.0;
    vertical_h[0][3] = 1.5;
    vertical_h[0][4] = 1.0;
    vertical_h[1][1] = 1.0;
    vertical_h[1][3] = -0.5;
    vertical_h[1][5] = 1.0;

    for (int t = 0; t < 2; t++) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 7; col++) {
                param_matrix_horizontal_h[t][row * 16 + col + 1 + row] = horizontal_h[t][col];
            }
        }
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 7; row++) {
                param_matrix_vertical_h[t][(row + col + 1) * 8 + col] = vertical_h[t][row];
            }
        }
    }

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8 * 16; i++) {
            param_matrix_horizontal_h[t][i] = param_matrix_horizontal_h[t][i] / 74;
            param_matrix_vertical_h[t][i] = param_matrix_vertical_h[t][i] / 74;
        }
    }

    // param vertical layout transformation
    for (int t = 0; t < 2; t++) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 4; row++) {
                param_matrix_vertical2_h[t][row * 8 + col] = param_matrix_vertical_h[t][row * 2 * 8 + col];
            }
            for (int row = 4; row < 8; row++) {
                param_matrix_vertical2_h[t][row * 8 + col] = param_matrix_vertical_h[t][(2 * row - 7) * 8 + col];
            }
            for (int row = 8; row < 12; row++) {
                param_matrix_vertical2_h[t][row * 8 + col] = param_matrix_vertical_h[t][(2 * row - 8) * 8 + col];
            }
            for (int row = 12; row < 16; row++) {
                param_matrix_vertical2_h[t][row * 8 + col] = param_matrix_vertical_h[t][(2 * row - 15) * 8 + col];
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_star2d1r_horizontal_d, param_matrix_horizontal_h,
                                  2 * 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_star2d1r_vertical_d, param_matrix_vertical2_h,
                                  2 * 8 * 16 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = rows * cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_M, BLOCK_N);

    dim3 block_config(32 * WARP_PER_BLOCK);

    // timing
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int i = 0;
    for (; i < times; i++) {
        kernel2d_star2d1r<<<grid_config, block_config>>>(array_d[i % 2],
                                                         array_d[(i + 1) % 2], cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(2D star_2d1r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_m * input_n * times * 3) / secs / 1e9);
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));


}

void gpu_star_2d3r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                   const int times, const int input_m, const int input_n) {
    double param_matrix_horizontal_h[16 * 8] = {0.0};
    double param_matrix_vertical_h[16 * 8] = {0.0};

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 7; col++) {
            param_matrix_horizontal_h[row * 16 + col + 1 + row] = params[col * 7 + 3];
        }
    }
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 7; row++) {
            if (row != 3) {
                param_matrix_vertical_h[(row + col + 1) * 8 + col] = params[3 * 7 + row];
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_star2d3r_horizontal_d, param_matrix_horizontal_h, 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_star2d3r_vertical_d, param_matrix_vertical_h, 8 * 16 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = rows * cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_M, BLOCK_N);

    dim3 block_config(32 * WARP_PER_BLOCK);

    // timing
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int i = 0;
    for (; i < times; i++) {
        kernel2d_star2d3r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2],
                                                                                cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(2D star_2d3r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_m * input_n * times) / secs / 1e9);
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

}

void gpu_box_2d3r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                             const int times, const int input_m, const int input_n) {
    double fact_param_matrix_h[4][49] = {0.0};
    double temp_param_matrix_h[3][49] = {0.0};
    // Factorize parameter matrix
    for (int i = -3; i < 1; i++) {
        if (i + 3 == 0) {
            for (int j = -3; j < 4; j++) {
                fact_param_matrix_h[0][(i + 3) * 7 + (j + 3)] = params[(i + 3) * 7 + (j + 3)];
                fact_param_matrix_h[0][(-i + 3) * 7 + (j + 3)] = params[(-i + 3) * 7 + (j + 3)];
            }
        } else {
            double prop = params[(i + 3) * 7] / params[0];
            for (int j = -3; j < 4; j++) {
                fact_param_matrix_h[0][(i + 3) * 7 + (j + 3)] = prop * params[(j + 3)];
                fact_param_matrix_h[0][(-i + 3) * 7 + (j + 3)] = fact_param_matrix_h[0][(i + 3) * 7 + (j + 3)];
                temp_param_matrix_h[0][(i + 3) * 7 + (j + 3)] =
                        params[(i + 3) * 7 + (j + 3)] - fact_param_matrix_h[0][(i + 3) * 7 + (j + 3)];
                temp_param_matrix_h[0][(-i + 3) * 7 + (j + 3)] = temp_param_matrix_h[0][(i + 3) * 7 + (j + 3)];
            }
        }
    }
    for (int i = -2; i < 1; i++) {
        if (i + 2 == 0) {
            for (int j = -2; j < 3; j++) {
                fact_param_matrix_h[1][(i + 3) * 7 + (j + 3)] = temp_param_matrix_h[0][(i + 3) * 7 + (j + 3)];
                fact_param_matrix_h[1][(-i + 3) * 7 + (j + 3)] = fact_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
            }
        } else {
            double prop = temp_param_matrix_h[0][(i + 3) * 7 + 1] / temp_param_matrix_h[0][1 * 7 + 1];
            for (int j = -2; j < 3; j++) {
                fact_param_matrix_h[1][(i + 3) * 7 + (j + 3)] = prop * temp_param_matrix_h[0][1 * 7 + (j + 3)];
                fact_param_matrix_h[1][(-i + 3) * 7 + (j + 3)] = fact_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
                temp_param_matrix_h[1][(i + 3) * 7 + (j + 3)] =
                        temp_param_matrix_h[0][(i + 3) * 7 + (j + 3)] - fact_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
                temp_param_matrix_h[1][(-i + 3) * 7 + (j + 3)] = temp_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
            }
        }
    }
    for (int i = -1; i < 1; i++) {
        if (i + 1 == 0) {
            for (int j = -1; j < 2; j++) {
                fact_param_matrix_h[2][(i + 3) * 7 + (j + 3)] = temp_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
                fact_param_matrix_h[2][(-i + 3) * 7 + (j + 3)] = temp_param_matrix_h[1][(i + 3) * 7 + (j + 3)];
            }
        } else {
            double prop = temp_param_matrix_h[1][(i + 3) * 7 + 2] / temp_param_matrix_h[1][2 * 7 + 2];
            for (int j = -1; j < 2; j++) {
                fact_param_matrix_h[2][(i + 3) * 7 + (j + 3)] = prop * temp_param_matrix_h[1][2 * 7 + (j + 3)];
                fact_param_matrix_h[2][(-i + 3) * 7 + (j + 3)] = fact_param_matrix_h[2][(i + 3) * 7 + (j + 3)];
                fact_param_matrix_h[3][(i + 3) * 7 + (j + 3)] =
                        temp_param_matrix_h[1][(i + 3) * 7 + (j + 3)] - fact_param_matrix_h[2][(i + 3) * 7 + (j + 3)];
            }
        }
    }

    double fact_param_matrix_u_h[4][7] = {0.0};
    double fact_param_matrix_v_h[4][7] = {0.0};

    for (int i = 0; i < 7; i++) {
        fact_param_matrix_u_h[0][i] = fact_param_matrix_h[0][i];
        fact_param_matrix_v_h[0][i] = fact_param_matrix_h[0][i * 7] / fact_param_matrix_h[0][0];
    }
    for (int i = 1; i < 6; i++) {
        fact_param_matrix_u_h[1][i] = fact_param_matrix_h[1][7 + i];
        fact_param_matrix_v_h[1][i] = fact_param_matrix_h[1][i * 7 + 1] / fact_param_matrix_h[1][7 + 1];
    }
    for (int i = 2; i < 5; i++) {
        fact_param_matrix_u_h[2][i] = fact_param_matrix_h[2][2 * 7 + i];
        fact_param_matrix_v_h[2][i] = fact_param_matrix_h[2][i * 7 + 2] / fact_param_matrix_h[2][2 * 7 + 2];
    }
    fact_param_matrix_u_h[3][3] = 1.0;
    fact_param_matrix_v_h[3][3] = fact_param_matrix_h[3][3 * 7 + 3];

    double param_matrix_u_h[4][16 * 8] = {0.0};
    double param_matrix_v_h[4][16 * 8] = {0.0};
    double param_matrix_u2_h[4][16 * 8] = {0.0};

    // Initialize parameter matrix
    for (int t = 0; t < 4; t++) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 7; row++) {
                param_matrix_u_h[t][(row + col + 1) * 8 + col] = fact_param_matrix_u_h[t][row];
            }
        }
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 7; col++) {
                param_matrix_v_h[t][row * 16 + col + 1 + row] = fact_param_matrix_v_h[t][col];
            }
        }
    }

    // param u layout transformation
    for (int t = 0; t < 4; t++) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 4; row++) {
                param_matrix_u2_h[t][row * 8 + col] = param_matrix_u_h[t][row * 2 * 8 + col];
            }
            for (int row = 4; row < 8; row++) {
                param_matrix_u2_h[t][row * 8 + col] = param_matrix_u_h[t][(2 * row - 7) * 8 + col];
            }
            for (int row = 8; row < 12; row++) {
                param_matrix_u2_h[t][row * 8 + col] = param_matrix_u_h[t][(2 * row - 8) * 8 + col];
            }
            for (int row = 12; row < 16; row++) {
                param_matrix_u2_h[t][row * 8 + col] = param_matrix_u_h[t][(2 * row - 15) * 8 + col];
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_u_d, param_matrix_u2_h, 4 * 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_v_d, param_matrix_v_h, 4 * 8 * 16 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = rows * cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_M, BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // timing
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int i = 0;
    for (; i < times; i++) {
//        breakdown1<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
//        breakdown2<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
//        breakdown3<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
        kernel2d_box2d3r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(2D box_2d3r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_m * input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));


}
