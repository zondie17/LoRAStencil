#include <iostream>
#include <omp.h>
#include "2d_utils.h"

const char *ShapeStr[5] = {
        "star_2d1r",
        "box_2d1r",
        "star_2d3r",
        "box_2d3r",
};

// Fill the matrix with random numbers or indices
#define FILL_RANDOM
// #define FILL_INDEX

/* Global variable */
int NY;
int XSLOPE, YSLOPE;

void printHelp() {
    const char *helpMessage =
            "Program name: lorastencil_2d\n"
            "Usage: lorastencil_2d shape input_size_of_first_dimension input_size_of_second_dimension time_size\n"
            "Shape: box2d1r or star2d1r or box2d3r or star2d3r\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printHelp();
        return 1;
    }

    // configurable settings
    Shape compute_shape;
    std::string arg1 = argv[1];
    if (arg1 == "box2d1r") {
        compute_shape = box_2d1r;
    } else if (arg1 == "star2d1r") {
        compute_shape = star_2d1r;
    } else if (arg1 == "star2d3r") {
        compute_shape = star_2d3r;
    } else if (arg1 == "box2d3r") {
        compute_shape = box_2d3r;
    } else {
        printHelp();
        return 1;
    }

    int m = 0;
    int n = 0;
    int times = 0;

    try {
        m = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        times = std::stoi(argv[4]);
    }
    catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e) {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    double param_1r[9] = {0.0};
    bool breakdown = false;

    double param_box_2d1r[49] = {0.0};
    double param_star_2d1r[49] = {0.0};

    for (int i = 0; i < 49; i++) {
        param_box_2d1r[i] = 0.021;
    }

    for (double &i: param_box_2d1r) {
        i = 0;
    }
    int param_num = 1;
    for (int i = -3; i < 1; i++) {
        for (int j = -3; j < 1; j++) {
            if (i <= j) {
                param_box_2d1r[(i + 3) * 7 + (j + 3)] = param_num;
                param_box_2d1r[(-i + 3) * 7 + (j + 3)] = param_num;
                param_box_2d1r[(i + 3) * 7 + (-j + 3)] = param_num;
                param_box_2d1r[(-i + 3) * 7 + (-j + 3)] = param_num;
                param_box_2d1r[(j + 3) * 7 + (i + 3)] = param_num;
                param_box_2d1r[(-j + 3) * 7 + (i + 3)] = param_num;
                param_box_2d1r[(j + 3) * 7 + (-i + 3)] = param_num;
                param_box_2d1r[(-j + 3) * 7 + (-i + 3)] = param_num;
                param_num++;
            }
        }
    }

    double *param;
    int halo;
    switch (compute_shape) {
        case box_2d1r:
            param = param_box_2d1r;
            halo = 3;
            break;
        case star_2d1r:
            param = param_star_2d1r;
            halo = 3;
            break;
        case star_2d3r:
            param = param_star_2d1r;
            halo = 3;
            break;
        case box_2d3r:
            param = param_box_2d1r;
            halo = 3;
            break;
    }

    // print brief info
    printf("INFO: shape = %s, m = %d, n = %d, times = %d\n", ShapeStr[compute_shape], m, n, times);

    int rows = m + 2 * 4;
    int cols = n + 2 * 4;
    NY = n;
    size_t matrix_size = (unsigned long) rows * cols * sizeof(double);

    // allocate space

    double *matrix = (double *) malloc(matrix_size);
    double *output = (double *) malloc(matrix_size);

    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double) (rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            matrix[i * cols + j] = (double)(i * (cols - 2) + j);
        }
    }
#else
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            matrix[i * cols + j] = 1.0;
        }
    }
    // std::fill_n(matrix, rows * cols, 1.0);
#endif

    switch (compute_shape) {
        case star_2d1r:
            gpu_star_2d1r(matrix, output, param, times, m, n);
            break;
        case star_2d3r: {
            for (int i = -3; i < 4; i++) {
                for (int j = -3; j < 4; j++) {
                    if (i != 0 && j != 0) {
                        param_box_2d1r[(i + 3) * 7 + (j + 3)] = 0;
                    }
                }
            }
            double param_total = 0;
            for (double i: param_box_2d1r)
                param_total += i;
            for (double &i: param_box_2d1r) {
                i = i / param_total;
            }
            param = param_box_2d1r;

            gpu_star_2d3r(matrix, output, param, times, m, n);
            break;
        }
        case box_2d1r:
        case box_2d3r:
            double param_total = 0;
            for (double i: param_box_2d1r)
                param_total += i;
            for (double &i: param_box_2d1r) {
                i = i / param_total;
            }
            param = param_box_2d1r;
            gpu_box_2d3r(matrix, output, param, times, m, n);
            break;
    }

    // free space
    free(output);
    free(matrix);

    return 0;

}