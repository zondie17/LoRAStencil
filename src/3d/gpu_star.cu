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

__constant__ double param_matrix_7p_horizontal_d[12 * 8];
__constant__ double param_matrix_7p_vertical_d[12 * 8];


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

__forceinline__ __device__ void compute_one_point(double *__restrict__ data, double *__restrict__ out) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
#pragma unroll
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[i] = 1 * data[IDX2D(row + ROW_HALO, col + HALO, D_BLOCK_SIZE_COL)];
    }
    __syncthreads();
}

__forceinline__ __device__ void
compute_tensorcore(double *__restrict__ data, double *__restrict__ out, const int ldm, const int warp_id) {
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> param_frag1;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag2;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> in_frag1;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag2;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0);

    // part1
#pragma unroll
    for (int row = 0; row < 3; row++) {
        wmma::load_matrix_sync(param_frag1, param_matrix_7p_horizontal_d + row * 4, 12);
        wmma::load_matrix_sync(in_frag1, data + 8 * warp_id + IDX2D(4 * row, 4, D_BLOCK_SIZE_COL),
                               D_BLOCK_SIZE_COL);
        wmma::mma_sync(acc_frag, param_frag1, in_frag1, acc_frag);
    }
#pragma unroll
    for (int row = 0; row < 3; row++) {
        wmma::load_matrix_sync(param_frag2, param_matrix_7p_vertical_d + row * 32, 8);
        wmma::load_matrix_sync(in_frag2, data + 8 * warp_id + IDX2D(2, 4 * row + 2, D_BLOCK_SIZE_COL),
                               D_BLOCK_SIZE_COL);
        wmma::mma_sync(acc_frag, in_frag2, param_frag2, acc_frag);
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
kernel_star3d1r(const double *__restrict__ in, double *__restrict__ out, const int heights, const int rows,
                const int cols) {
    __shared__ double sharedmem[
            D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + 4 * BLOCK_SIZE_COL * BLOCK_SIZE_ROW];

    int begin = IDX2D(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, cols);
    int warp_idx = threadIdx.x / 32;

    load_shared_data(sharedmem, in, 0, rows, cols);
    compute_one_point(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW);
    load_shared_data(sharedmem, in, 1, rows, cols);
    compute_one_point(sharedmem,
                          sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_tensorcore(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                                      3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, D_BLOCK_SIZE_COL, warp_idx);
#pragma unroll
    for (int h = 2; h < heights + 2; h++) {
        load_shared_data(sharedmem, in, h, rows, cols);
        compute_one_point(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                                         (h % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
        add(sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                ((h - 2) % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
                sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
                sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW + (h % 3) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL,
                out + (h - 1) * rows * cols + begin + IDX2D(ROW_HALO, HALO, cols), cols);
        compute_tensorcore(sharedmem, sharedmem + D_BLOCK_SIZE_COL * D_BLOCK_SIZE_ROW +
                                          3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, D_BLOCK_SIZE_COL,
                               warp_idx);
    }

}


void gpu_star_3d1r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params,
                   const int times, const int input_h, const int input_m, const int input_n) {
    double param_matrix_horizontal_h[12 * 8] = {0.0};
    double param_matrix_vertical_h[12 * 8] = {0.0};

    // Initialize parameter matrix
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 3; col++) {
            param_matrix_horizontal_h[row * 12 + col + 1 + row] = 1;
        }
    }
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 3; row++) {
            param_matrix_vertical_h[(row + col + 1) * 8 + col] = 1;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_7p_horizontal_d, param_matrix_horizontal_h,
                                  12 * 8 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_7p_vertical_d, param_matrix_vertical_h,
                                  12 * 8 * sizeof(double)));

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
        CUDAKERNELCHECK(
                (kernel_star3d1r<<<grid_config, block_config>>>
                        (array_d[i % 2], array_d[(i + 1) % 2], input_h, rows, cols)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(3D star_3d1r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
              << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_m * input_n * input_h * times) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));


}
