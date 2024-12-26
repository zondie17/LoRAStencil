#include <mma.h>
#include <iostream>
#include "1d_utils.h"
#include <chrono>

using namespace nvcuda;

#define HALO 4
#define WARP_PER_BLOCK 8
#define MATRIX_PER_WARP 4
#define FRAGMENT_M 8
#define BLOCK_COL 4 * WARP_PER_BLOCK * MATRIX_PER_WARP
#define BLOCK_SIZE (BLOCK_COL * FRAGMENT_M)
#define D_BLOCK_COL (BLOCK_COL + 2 * HALO)
#define D_BLOCK_SIZE (D_BLOCK_COL * FRAGMENT_M)


#define IDX(x, y, ldm) ((x) * (ldm) + (y))

__constant__ double param_matrix_d[16 * 8];

__global__ void kernel_1d2r(const double *__restrict__ in, double *__restrict__ out) {
    __shared__ double sharedmem[D_BLOCK_SIZE];
    int begin = blockIdx.x * BLOCK_SIZE;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE; i += totalThreads) {
        int row = i / D_BLOCK_COL;
        int col = i % D_BLOCK_COL;

        int base_addr = __cvta_generic_to_shared(sharedmem);
        int dst = base_addr + i * sizeof(double);
        asm ("cp.async.ca.shared.global [%0], [%1], 8;\n" :
            : "r"(dst), "l"(&in[begin + IDX(row, col, BLOCK_COL)]));
    }
    asm ("cp.async.commit_group;\n"::);
    asm ("cp.async.wait_group 0;\n"::);

    __syncthreads();

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::load_matrix_sync(param_frag[i], param_matrix_d + i * 32, 8);
    }

    nvcuda::wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag[2];
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;

    int warp_id = threadIdx.x / 32;
    int warp_begin = warp_id * 4 * MATRIX_PER_WARP;

    nvcuda::wmma::fill_fragment(acc_frag[0], 0.0);
    nvcuda::wmma::fill_fragment(acc_frag[1], 0.0);
    int data = 0;

#pragma unroll
    for (int row = 0; row < 2; row++) {
        wmma::load_matrix_sync(in_frag, sharedmem + warp_begin + 4 * row, D_BLOCK_COL);
        wmma::mma_sync(acc_frag[0], in_frag, param_frag[row], acc_frag[0]);
    }

#pragma unroll
    for (; data < (MATRIX_PER_WARP / 2) - 1; data++) {
#pragma unroll
        for (int row = 2; row < 4; row++) {
            wmma::load_matrix_sync(in_frag, sharedmem + warp_begin + 8 * data + 4 * row, D_BLOCK_COL);
            wmma::mma_sync(acc_frag[data % 2], in_frag, param_frag[row], acc_frag[data % 2]);
            wmma::mma_sync(acc_frag[(data + 1) % 2], in_frag, param_frag[row - 2], acc_frag[(data + 1) % 2]);
        }

        wmma::store_matrix_sync(out + begin + warp_begin + IDX(data, 4, 8), acc_frag[data % 2], BLOCK_COL,
                                wmma::mem_row_major);
        nvcuda::wmma::fill_fragment(acc_frag[data % 2], 0.0);
    }

#pragma unroll
    for (int row = 2; row < 4; row++) {
        wmma::load_matrix_sync(in_frag, sharedmem + warp_begin + 8 * data + 4 * row, D_BLOCK_COL);
        wmma::mma_sync(acc_frag[data % 2], in_frag, param_frag[row], acc_frag[data % 2]);
    }
    wmma::store_matrix_sync(out + begin + warp_begin + IDX(data, 4, 8), acc_frag[data % 2], BLOCK_COL,
                            wmma::mem_row_major);
}


void
gpu_1d2r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times,
         const int input_n) {
    double param_matrix_h[8 * 16] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 9; row++) {
            param_matrix_h[(row + col) * 8 + col] = params[row];
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 16 * 8 * sizeof(double)));

    const int cols = input_n + 2 * HALO;
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++) {
        kernel_1d2r<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "LoRAStencil(1D 1d2r): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
            << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double) input_n * times * 2) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size - sizeof(double), cudaMemcpyDeviceToHost));
}
