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
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2 + 4 )
#define D_BLOCK_SIZE_COL_NOPAD (BLOCK_SIZE_COL + HALO * 2)
#define TENSOR_CORE_M 8
#define TENSOR_CORE_N 4
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 8
#define CALC_NUM 4
#define DATA_NUM 5

__constant__ double matrix_U_d[3 * 16 * 8];
__constant__ double matrix_V_d[3 * 8 * 16];

__constant__ double matrix_star2d3r_U_d[16 * 8];
__constant__ double matrix_star2d3r_V_d[8 * 16];

__constant__ double matrix_star2d1r_U_d[16 * 8];
__constant__ double matrix_star2d1r_V_d[8 * 16];

__global__ void
kernel2d_box2d3r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll 4
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL_NOPAD / 2; i += totalThreads) {
        int row = (i * 2) / D_BLOCK_SIZE_COL_NOPAD;
        int col = (i * 2) % D_BLOCK_SIZE_COL_NOPAD;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + (row * D_BLOCK_SIZE_COL + col) * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
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

    wmma::fill_fragment(acc_frag[0], 0.0);
    wmma::fill_fragment(acc_frag[1], 0.0);
    wmma::fill_fragment(acc_frag[2], 0.0);
    wmma::fill_fragment(acc_frag[3], 0.0);

#pragma unroll 2
    for (int param = 0; param < 3; param++) {

        wmma::fill_fragment(acc_v_frag[0], 0.0);
        wmma::fill_fragment(acc_v_frag[1], 0.0);
        wmma::fill_fragment(acc_v_frag[2], 0.0);
        wmma::fill_fragment(acc_v_frag[3], 0.0);
        wmma::fill_fragment(acc_v_frag[4], 0.0);

        // part1
#pragma unroll 2
        for (int row = 0; row < 4; row++) {
            wmma::load_matrix_sync(param_frag1, matrix_U_d + param * 128 + row * 4, 16);
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
#pragma unroll 4
        for (int row = 0; row < 4; row++) {
            wmma::load_matrix_sync(param_frag2, matrix_V_d + param * 128 + row * 32, 8);
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

__global__ void
kernel2d_star2d3r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll 4
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL_NOPAD / 2; i += totalThreads) {
        int row = (i * 2) / D_BLOCK_SIZE_COL_NOPAD;
        int col = (i * 2) % D_BLOCK_SIZE_COL_NOPAD;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + (row * D_BLOCK_SIZE_COL + col) * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
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

    wmma::fill_fragment(acc_frag[0], 0.0);
    wmma::fill_fragment(acc_frag[1], 0.0);
    wmma::fill_fragment(acc_frag[2], 0.0);
    wmma::fill_fragment(acc_frag[3], 0.0);

    // part1
#pragma unroll
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag1, matrix_star2d3r_U_d + row * 4, 16);
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
        wmma::load_matrix_sync(param_frag2, matrix_star2d3r_V_d + row * 32, 8);
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
kernel2d_star2d1r(const double *__restrict__ in, double *__restrict__ out, const int ldm) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll 4
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL_NOPAD / 2; i += totalThreads) {
        int row = (i * 2) / D_BLOCK_SIZE_COL_NOPAD;
        int col = (i * 2) % D_BLOCK_SIZE_COL_NOPAD;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + (row * D_BLOCK_SIZE_COL + col) * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
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

    wmma::fill_fragment(acc_frag[0], 0.0);
    wmma::fill_fragment(acc_frag[1], 0.0);
    wmma::fill_fragment(acc_frag[2], 0.0);
    wmma::fill_fragment(acc_frag[3], 0.0);

    wmma::fill_fragment(acc_v_frag[0], 0.0);
    wmma::fill_fragment(acc_v_frag[1], 0.0);
    wmma::fill_fragment(acc_v_frag[2], 0.0);
    wmma::fill_fragment(acc_v_frag[3], 0.0);
    wmma::fill_fragment(acc_v_frag[4], 0.0);


    // part1
#pragma unroll 2
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag1, matrix_star2d1r_U_d + row * 4, 16);
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
#pragma unroll 4
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag2, matrix_star2d1r_V_d + row * 32, 8);
#pragma unroll
        for (int data = 0; data < 4; data++) {
            acc_temp_frag.x[0] = acc_v_frag[data + row / 2].x[row % 2];
            wmma::mma_sync(acc_frag[data], acc_temp_frag, param_frag2, acc_frag[data]);
        }
    }

    int row = (tid % 32) / 4;
    int col = ((tid % 32) % 4) * 2;
    for (int data = 0; data < 4; data++) {
        int data_begin = warp_begin + IDX(data, 4 * D_BLOCK_SIZE_COL + 4, 8);
        for (int num = 0; num < 2; num++) {
            acc_frag[data].x[num] = acc_frag[data].x[num] +
                sharedmem[data_begin + IDX(row, col + num - 3, D_BLOCK_SIZE_COL)] +
                sharedmem[data_begin + IDX(row, col + num + 3, D_BLOCK_SIZE_COL)] +
                sharedmem[data_begin + IDX(row - 3, col + num, D_BLOCK_SIZE_COL)] +
                sharedmem[data_begin + IDX(row + 3, col + num, D_BLOCK_SIZE_COL)] -
                sharedmem[data_begin + IDX(row - 2, col + num - 2, D_BLOCK_SIZE_COL)] -
                sharedmem[data_begin + IDX(row - 2, col + num + 2, D_BLOCK_SIZE_COL)] -
                sharedmem[data_begin + IDX(row + 2, col + num - 2, D_BLOCK_SIZE_COL)] -
                sharedmem[data_begin + IDX(row + 2, col + num + 2, D_BLOCK_SIZE_COL)];
        }
    }

#pragma unroll
    for (int data = 0; data < 4; data++) {
        wmma::store_matrix_sync(
                out + begin + IDX((warp_id / 2) * 8, (warp_id % 2) * 32, ldm) + IDX(data, 4 * ldm + 4, 8),
                acc_frag[data], ldm, wmma::mem_row_major);
    }

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


    double param_matrix_U[3][16*8] = {0.0};
    double param_matrix_V[3][16*8] = {0.0};
    double param_matrix_V2[3][16*8] = {0.0};

    // Initialize test param matrix
    for (int t = 0; t < 3; t++) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 7; col++) {
                param_matrix_U[t][row * 16 + col + 1 + row] = fact_param_matrix_u_h[t][col];
            }
        }
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 7; row++) {
                param_matrix_V[t][(row + col + 1) * 8 + col] = fact_param_matrix_v_h[t][row];
            }
        }
    }

    // test param V layout transformation
    for (int t = 0; t < 3; t++) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 4; row++) {
                param_matrix_V2[t][row * 8 + col] = param_matrix_V[t][row * 2 * 8 + col];
            }
            for (int row = 4; row < 8; row++) {
                param_matrix_V2[t][row * 8 + col] = param_matrix_V[t][(2 * row - 7) * 8 + col];
            }
            for (int row = 8; row < 12; row++) {
                param_matrix_V2[t][row * 8 + col] = param_matrix_V[t][(2 * row - 8) * 8 + col];
            }
            for (int row = 12; row < 16; row++) {
                param_matrix_V2[t][row * 8 + col] = param_matrix_V[t][(2 * row - 15) * 8 + col];
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(matrix_U_d, param_matrix_U, 3 * 16 * 8 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(matrix_V_d, param_matrix_V2, 3 * 16 * 8 * sizeof(double)));

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

void gpu_star_2d3r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                   const int times, const int input_m, const int input_n) {

    double param_matrix_U[16*8] = {0.0};
    double param_matrix_V[16*8] = {0.0};

    // Initialize test param matrix
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 7; col++) {
            param_matrix_U[row * 16 + col + 1 + row] = params[col * 7 + 3];
        }
    }
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 7; row++) {
            if (row != 3) {
                param_matrix_V[(row + col + 1) * 8 + col] = params[3 * 7 + row];
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d3r_U_d, param_matrix_U, 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d3r_V_d, param_matrix_V, 8 * 16 * sizeof(double)));


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
        kernel2d_star2d3r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
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


void gpu_star_2d1r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                   const int times, const int input_m, const int input_n) {
    double param_u[7] = {0, 1, 2, 4, 2, 1, 0};
    double param_v[7] = {0, 1, 2, 4, 2, 1, 0};

    double param_matrix_U[16*8] = {0.0};
    double param_matrix_V[16*8] = {0.0};
    double param_matrix_V2[16*8] = {0.0};

    // Initialize test param matrix
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 7; col++) {
            param_matrix_U[row * 16 + col + 1 + row] = param_u[col];
        }
    }
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 7; row++) {
            param_matrix_V[(row + col + 1) * 8 + col] = param_v[row];
        }
    }

    // test param V layout transformation
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 4; row++) {
            param_matrix_V2[row * 8 + col] = param_matrix_V[row * 2 * 8 + col];
        }
        for (int row = 4; row < 8; row++) {
            param_matrix_V2[row * 8 + col] = param_matrix_V[(2 * row - 7) * 8 + col];
        }
        for (int row = 8; row < 12; row++) {
            param_matrix_V2[row * 8 + col] = param_matrix_V[(2 * row - 8) * 8 + col];
        }
        for (int row = 12; row < 16; row++) {
            param_matrix_V2[row * 8 + col] = param_matrix_V[(2 * row - 15) * 8 + col];
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d1r_U_d, param_matrix_U, 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d1r_V_d, param_matrix_V2, 8 * 16 * sizeof(double)));


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
        kernel2d_star2d1r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols);
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
