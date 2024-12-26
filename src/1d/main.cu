#include <iostream>
#include <omp.h>
#include "1d_utils.h"
#include <cmath>

const char *ShapeStr[4] = {
    "1d1r",
    "1d2r"
};

// Fill the matrix with random numbers or indices
#define FILL_RANDOM
// #define FILL_INDEX

// Check the correctness of the results or not
// #define CHECK_ERROR
const double tolerance = 1e-7;

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define ABS(x, y) (((x) > (y)) ? ((x) - (y)): ((y) - (x)))


int HALO;


void printHelp() {
    const char *helpMessage =
            "Program name: lorastencil_1d\n"
            "Usage: lorastencil_1d shape input_size time_size\n"
            "Shape: 1d1r or 1d2r\n";
    printf("%s\n", helpMessage);
}

void test_cpu(double *in, double *out, double *param, const int input_n) {
    for (int col = 4; col < input_n - 4; col++) {
        out[col] =
                param[0] * in[col - 4] + param[1] * in[col - 3] + param[2] * in[col - 2] + param[3] * in[col - 1] +
                param[4] * in[col] + param[5] * in[col + 1] + param[6] * in[col + 2] + param[7] * in[col + 3] + param[8] * in[col + 4];
    }
}


int main(int argc, char *argv[]) {
    if (argc < 4) {
        printHelp();
        return 1;
    }

    // configurable settings
    std::string arg1 = argv[1];

    Shape compute_shape;
    if (arg1 == "1d1r") {
        compute_shape = star_1d1r;
    } else if (arg1 == "1d2r") {
        compute_shape = star_1d2r;
    } else {
        printHelp();
        return 1;
    }

    int n = 0;
    int times = 0;

    try {
        n = std::stoi(argv[2]);
        times = std::stoi(argv[3]);
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e) {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    double param_1d1r[9] = {0, 1, 2, 3, 4, 3, 2, 1, 0};
    double param_1d2r[9] = {1, 2, 3, 4, 5, 4, 3, 2, 1};

    double *param;

    switch (compute_shape) {
        case star_1d1r:
            param = param_1d1r;
            break;
        case star_1d2r:
            param = param_1d2r;
            break;
    }


    // print brief info

    printf("INFO: shape = %s, n = %d, times = %d\n", ShapeStr[compute_shape], n, times);

    int cols = n + 2 * 4;
    size_t matrix_size = (unsigned long) cols * sizeof(double);

    // allocate space
    double *matrix = (double *) malloc(matrix_size);
    double *output = (double *) malloc(matrix_size);

    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < cols + 1; i++) {
        matrix[i] = (double) (rand() % 10000);
    }
#elif defined(FILL_INDEX)
    for (int i = 4; i < cols - 4; i++) {
        matrix[i] = (double) (i - 4);
    }
#else
    std::fill_n(matrix, matrix_size, 1.0);
#endif

#if defined(CHECK_ERROR)
    std::cout << arg1 << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << param[i] << std::endl;
    }
#endif


    switch (compute_shape) {
        case star_1d1r:
            gpu_1d1r(matrix, output, param, times, n);
            break;
        case star_1d2r:
            gpu_1d2r(matrix, output, param, times, n);
            break;
    }

#if defined(CHECK_ERROR)
    printf("\nChecking Correctness ... \n");
    double *naive[2];
    naive[0] = (double *) malloc(matrix_size);
    naive[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < cols; i++) {
        naive[0][i] = matrix[i];
        naive[1][i] = 0;
    }

    double *lora[2];
    lora[0] = (double *) malloc(matrix_size);
    lora[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < cols; i++) {
        lora[0][i] = matrix[i];
        lora[1][i] = 0;
    }

    test_cpu(naive[0], naive[1], param, cols);
    switch (compute_shape) {
        case star_1d1r:
            gpu_1d1r(lora[0], lora[1], param, 1, n);
            break;
        case star_1d2r:
            gpu_1d2r(lora[0], lora[1], param, 1, n);
            break;
    }
    printf("Comparing naive and lora\n");
    for (int col = 0; col < cols - 4; col++) {
        if (ABS(naive[1][col], lora[1][col]) > 1e-7) {
            printf("col = %d, naive = %lf, lora = %lf\n", col, naive[1][col], lora[1][col]);
        }
    }

    printf("Correct!\n");

#endif


    // free space
    free(output);
    free(matrix);

    return 0;
}
