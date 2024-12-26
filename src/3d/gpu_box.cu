#include <mma.h>
#include <iostream>
#include "3d_utils.h"
#include <chrono>

using namespace nvcuda;

#define HALO 4
#define ROW_HALO 2
#define WARP_PER_BLOCK 8
#define BLOCK_SIZE_ROW 8
#define BLOCK_SIZE_COL 64
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + ROW_HALO * 2)
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define IDX2D(x, y, ldm) ((x) * (ldm) + (y))
#define IDX3D(x, y, z, rows, cols) ((x) * (rows) * (cols) + (y) * (cols) + (z))

__constant__ double param_matrix_27p_horizontal_d[3 * 12 * 8];
__constant__ double param_matrix_27p_vertical_d[3 * 16 * 8];

__forceinline__ __device__ void
load_shared_data(double *__restrict__ sharedmem, const double *__restrict__ in, const int h, const int rows,
                 const int cols) {
    int begin = IDX3D(h, blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, rows, cols);
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += total_threads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(dst), "l"(&in[begin + IDX2D(row, col, cols)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);
    __syncthreads();
}


__forceinline__ __device__ void
compute_tensorcore(double *__restrict__ data, double *__restrict__ out, const int ldm, const int warp_id,
                   const int param_idx) {
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1[3];
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2[4];
#pragma unroll
    for (int row = 0; row < 3; row++) {
        wmma::load_matrix_sync(param_frag1[row], param_matrix_27p_horizontal_d + param_idx * 96 + row * 4, 12);
    }
#pragma unroll
    for (int row = 0; row < 4; row++) {
        wmma::load_matrix_sync(param_frag2[row], param_matrix_27p_vertical_d + param_idx * 128 + row * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_v_frag[2];
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> acc_temp_frag;


    wmma::fill_fragment(acc_frag, 0.0);
    wmma::fill_fragment(acc_v_frag[0], 0.0);
    wmma::fill_fragment(acc_v_frag[1], 0.0);
    // part1
#pragma unroll
    for (int row = 0; row < 3; row++) {
#pragma unroll
        for (int num = 0; num < 2; num++) {
            wmma::load_matrix_sync(in_frag, data + IDX2D(4 * row, 8 * warp_id + 8 * num, D_BLOCK_SIZE_COL),
                                   D_BLOCK_SIZE_COL);
            wmma::mma_sync(acc_v_frag[num], param_frag1[row], in_frag, acc_v_frag[num]);
        }
    }

    // part2
#pragma unroll
    for (int row = 0; row < 4; row++) {
        acc_temp_frag.x[0] = acc_v_frag[row / 2].x[row % 2];
        wmma::mma_sync(acc_frag, acc_temp_frag, param_frag2[row], acc_frag);
    }
    wmma::store_matrix_sync(out + warp_id * 8, acc_frag, BLOCK_SIZE_COL, wmma::mem_row_major);

    __syncthreads();
}


__forceinline__ __device__ void
add(double *__restrict__ data1, double *__restrict__ data2, double *__restrict__ data3, double *__restrict__ out,
    const int cols) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
#pragma unroll
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[IDX2D(row, col, cols)] = data1[i] + data2[i] + data3[i];
    }

    __syncthreads();
}


__global__ void
kernel_box3d1r(const double *__restrict__ in, double *__restrict__ out, const int heights, const int rows,
               const int cols) {
    __shared__ double sharedmem[D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + 4 * BLOCK_SIZE_COL * BLOCK_SIZE_ROW];


    int begin = IDX2D(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, cols);
    int warp_idx = threadIdx.x / 32;

    load_shared_data(sharedmem, in, 0, rows, cols);
    compute_tensorcore(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW, D_BLOCK_SIZE_COL,
                       warp_idx, 0);
    load_shared_data(sharedmem, in, 1, rows, cols);
    compute_tensorcore(sharedmem,
                       sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
                       D_BLOCK_SIZE_COL, warp_idx, 0);
    compute_tensorcore(sharedmem,
                       sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                       3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
                       D_BLOCK_SIZE_COL, warp_idx, 1);
#pragma unroll
    for (int h = 2; h < heights + 2; h++) {
        load_shared_data(sharedmem, in, h, rows, cols);
        compute_tensorcore(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                                      (h % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, D_BLOCK_SIZE_COL, warp_idx, 0);
        add(sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
            ((h - 2) % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
            sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
            3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
            sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + (h % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
            out + (h - 1) * rows * cols + begin + IDX2D(ROW_HALO, HALO, cols), cols);
        compute_tensorcore(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                                      3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, D_BLOCK_SIZE_COL,
                           warp_idx, 1);
    }
}


void gpu_box_3d1r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                  const int times, const int input_h, const int input_m, const int input_n) {
    double param_matrix_horizontal_h[3][12 * 8] = {0.0};
    double param_matrix_vertical_h[3][16 * 8] = {0.0};
    double param_matrix_v_h[3][16 * 8] = {0.0};


    // Initialize parameter matrix
    for (auto &t: param_matrix_horizontal_h) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 3; col++) {
                t[row * 12 + col + 1 + row] = 1;
            }
        }
    }
    for (auto &t: param_matrix_v_h) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 3; row++) {
                t[(row + col + 3) * 8 + col] = params[row % 3];
            }
        }
    }

    // layout transformation for vertical matrix
    for (int t = 0; t < 3; t++) {
        for (int col = 0; col < 8; col++) {
            for (int row = 0; row < 4; row++) {
                param_matrix_vertical_h[t][row * 8 + col] = param_matrix_v_h[t][row * 2 * 8 + col];
            }
            for (int row = 4; row < 8; row++) {
                param_matrix_vertical_h[t][row * 8 + col] = param_matrix_v_h[t][(2 * row - 7) * 8 + col];
            }
            for (int row = 8; row < 12; row++) {
                param_matrix_vertical_h[t][row * 8 + col] = param_matrix_v_h[t][(2 * row - 8) * 8 + col];
            }
            for (int row = 12; row < 16; row++) {
                param_matrix_vertical_h[t][row * 8 + col] = param_matrix_v_h[t][(2 * row - 15) * 8 + col];
            }
        }
    }


    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_27p_horizontal_d, param_matrix_horizontal_h,
                                  3 * 12 * 8 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_27p_vertical_d, param_matrix_vertical_h,
                                  3 * 16 * 8 * sizeof(double)));

    const int heights = input_h + 2 * 1;
    const int rows = input_m + 2 * ROW_HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = heights * rows * cols * sizeof(double);
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
        CUDAKERNELCHECK((kernel_box3d1r<<<grid_config, block_config>>>
                (array_d[i % 2], array_d[(i + 1) % 2], input_h, rows, cols)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(3D box_3d1r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_m * input_n * input_h * times) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));


}

