#include <iostream>
#include <omp.h>
#include "3d_utils.h"

const char *ShapeStr[5] = {
    "box_3d1r",
    "star_3d1r",
};

#define FILL_RANDOM
// #define FILL_INDEX

// Check the correctness of the results or not
// #define CHECK_ERROR
const double tolerance = 1e-7;

#define IDX2D(x, y, ldm) ((x) * (ldm) + (y))
#define IDX3D(x, y, z, rows, cols) ((x) * (rows) * (cols) + (y) * (cols) + (z))
#define ABS(x, y) (((x) > (y)) ? ((x) - (y)): ((y) - (x)))

#define HEIGHT_HALO 1
#define ROW_HALO 2
#define COL_HALO 4

void printHelp() {
    const char *helpMessage =
            "Program name: lorastencil_3d\n"
            "Usage: lorastencil_3d shape input_size_of_first_dimension input_size_of_second_dimension input_size_of_third_dimension time_size\n"
            "Shape: box3d1r or star3d1r\n";
    printf("%s\n", helpMessage);
}

void test_cpu(double *in, double *out, double *param, const int heights, const int rows, const int cols) {
    for (int height = HEIGHT_HALO; height < heights - HEIGHT_HALO; height++) {
        for (int row = ROW_HALO; row < rows - ROW_HALO; row++) {
            for (int col = COL_HALO; col < cols - COL_HALO; col++) {
                out[IDX3D(height, row, col, rows, cols)] =
                        param[0] * in[IDX3D(height - 1, row - 1, col - 1, rows, cols)] +
                        param[1] * in[IDX3D(height - 1, row - 1, col, rows, cols)] +
                        param[2] * in[IDX3D(height - 1, row - 1, col + 1, rows, cols)] +
                        param[3] * in[IDX3D(height - 1, row, col - 1, rows, cols)] +
                        param[4] * in[IDX3D(height - 1, row, col, rows, cols)] +
                        param[5] * in[IDX3D(height - 1, row, col + 1, rows, cols)] +
                        param[6] * in[IDX3D(height - 1, row + 1, col - 1, rows, cols)] +
                        param[7] * in[IDX3D(height - 1, row + 1, col, rows, cols)] +
                        param[8] * in[IDX3D(height - 1, row + 1, col + 1, rows, cols)] +
                        param[9] * in[IDX3D(height, row - 1, col - 1, rows, cols)] +
                        param[10] * in[IDX3D(height, row - 1, col, rows, cols)] +
                        param[11] * in[IDX3D(height, row - 1, col + 1, rows, cols)] +
                        param[12] * in[IDX3D(height, row, col - 1, rows, cols)] +
                        param[13] * in[IDX3D(height, row, col, rows, cols)] +
                        param[14] * in[IDX3D(height, row, col + 1, rows, cols)] +
                        param[15] * in[IDX3D(height, row + 1, col - 1, rows, cols)] +
                        param[16] * in[IDX3D(height, row + 1, col, rows, cols)] +
                        param[17] * in[IDX3D(height, row + 1, col + 1, rows, cols)] +
                        param[18] * in[IDX3D(height + 1, row - 1, col - 1, rows, cols)] +
                        param[19] * in[IDX3D(height + 1, row - 1, col, rows, cols)] +
                        param[20] * in[IDX3D(height + 1, row - 1, col + 1, rows, cols)] +
                        param[21] * in[IDX3D(height + 1, row, col - 1, rows, cols)] +
                        param[22] * in[IDX3D(height + 1, row, col, rows, cols)] +
                        param[23] * in[IDX3D(height + 1, row, col + 1, rows, cols)] +
                        param[24] * in[IDX3D(height + 1, row + 1, col - 1, rows, cols)] +
                        param[25] * in[IDX3D(height + 1, row + 1, col, rows, cols)] +
                        param[26] * in[IDX3D(height + 1, row + 1, col + 1, rows, cols)];
            }
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc < 6) {
        printHelp();
        return 1;
    }

    // configurable settings
    Shape compute_shape;
    std::string arg1 = argv[1];
    if (arg1 == "box3d1r") {
        compute_shape = box_3d1r;
    } else if (arg1 == "star3d1r") {
        compute_shape = star_3d1r;
    } else {
        printHelp();
        return 1;
    }

    int h = 0;
    int m = 0;
    int n = 0;
    int times = 0;

    try {
        h = std::stoi(argv[2]);
        m = std::stoi(argv[3]);
        n = std::stoi(argv[4]);
        times = std::stoi(argv[5]);
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e) {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    // print brief info

    printf("INFO: shape = %s, h = %d, m = %d, n = %d, times = %d\n", ShapeStr[compute_shape], h, m, n, times);

    double param_box_3d1r[27] = {0};

    param_box_3d1r[0] = 1;
    param_box_3d1r[1] = 2;
    param_box_3d1r[2] = 1;
    for (int i = 3; i < 27; i++) {
        param_box_3d1r[i] = param_box_3d1r[i % 3];
    }

    double param_star_3d1r[27] = {
        0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 2, 1, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0
    };


    double *param;
    switch (compute_shape) {
        case box_3d1r:
            param = param_box_3d1r;
            break;
        case star_3d1r:
            param = param_star_3d1r;
            break;
    }


#ifdef CHECK_ERROR
    std::cout << arg1 << std::endl;
    for (int height = 0; height < 3; height++) {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                std::cout << param[IDX3D(height, row, col, 3, 3)] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

    int heights = h + 2 * HEIGHT_HALO;
    int rows = m + 2 * ROW_HALO;
    int cols = n + 2 * COL_HALO;
    size_t matrix_size = (unsigned long) heights * rows * cols * sizeof(double);

    double *matrix = (double *) malloc(matrix_size);
    double *output = (double *) malloc(matrix_size);

    for (int i = 0; i < heights * rows * cols; i++) {
        matrix[i] = 0;
    }

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < heights * rows * cols; i++) {
        matrix[i] = (double) (rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int height = HEIGHT_HALO; height < heights - HEIGHT_HALO; height++) {
        for (int row = ROW_HALO; row < rows - ROW_HALO; row++) {
            for (int col = COL_HALO; col < cols - COL_HALO; col++) {
                matrix[IDX3D(height, row, col, rows, cols)] = (double) IDX3D(
                    height - HEIGHT_HALO, row - ROW_HALO, col - COL_HALO, rows - 2 * ROW_HALO, cols - 2 * COL_HALO);
            }
        }
    }
    // for (int height = HEIGHT_HALO; height < heights - HEIGHT_HALO; height++) {
    //     for (int row = ROW_HALO; row < rows - ROW_HALO; row++) {
    //         for (int col = COL_HALO; col < cols - COL_HALO; col++) {
    //             std::cout << matrix[IDX3D(height, row, col, rows, cols)] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
#else
    std::fill_n(matrix, heights * rows * cols, 1.0);
#endif


    switch (compute_shape) {
        case box_3d1r:
            gpu_box_3d1r(matrix, output, param, times, h, m, n);
            break;
        case star_3d1r:
            gpu_star_3d1r(matrix, output, param, times, h, m, n);
            break;
    }

#if defined(CHECK_ERROR)
    printf("\nChecking Correctness... \n");
    double *naive[2];
    naive[0] = (double *) malloc(matrix_size);
    naive[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < heights * rows * cols; i++) {
        naive[0][i] = matrix[i];
        naive[1][i] = 0;
    }

    double *lora[2];
    lora[0] = (double *) malloc(matrix_size);
    lora[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < heights * rows * cols; i++) {
        lora[0][i] = matrix[i];
        lora[1][i] = 0;
    }


    test_cpu(naive[0], naive[1], param, heights, rows, cols);
    switch (compute_shape) {
        case box_3d1r:
            gpu_box_3d1r(lora[0], lora[1], param, 1, h, m, n);
        break;
        case star_3d1r:
            gpu_star_3d1r(lora[0], lora[1], param, 1, h, m, n);
        break;
    }

    printf("Comparing naive and lora\n");
    for (int height = HEIGHT_HALO; height < heights - HEIGHT_HALO; height++) {
        for (int row = ROW_HALO; row < rows - ROW_HALO; row++) {
            for (int col = COL_HALO; col < cols - COL_HALO; col++) {
                if (ABS(naive[1][IDX3D(height, row, col, rows, cols)],
                        lora[1][IDX3D(height, row, col, rows, cols)]) > 1e-7) {
                    printf("height = %d, row = %d, col = %d, naive = %lf, output = %lf\n", height, row, col,
                           naive[1][IDX3D(height, row, col, rows, cols)],
                           lora[1][IDX3D(height, row, col, rows, cols)]);
                }
            }
        }
    }
    printf("Correct!\n");
#endif

    // free space
    free(output);
    free(matrix);

    return 0;
}
