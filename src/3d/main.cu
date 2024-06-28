#include <iostream>
#include <omp.h>
#include "3d_utils.h"

const char *ShapeStr[5] = {
        "box_3d1r",
        "star_3d1r",
};

#define FILL_RANDOM
// #define FILL_INDEX

void printHelp() {
    const char *helpMessage =
            "Program name: lorastencil_3d\n"
            "Usage: lorastencil_3d shape input_size_of_first_dimension input_size_of_second_dimension input_size_of_third_dimension time_size\n"
            "Shape: box3d1r or star3d1r\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[])  {
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
    }
    catch (const std::invalid_argument &e) {
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

    double *param = param_box_3d1r;

    int _h = h + 2 * 1;
    int _r = m + 2 * 2;
    int _c = n + 2 * 4;
    size_t matrix_size = (unsigned long) _h * _r * _c * sizeof(double);

    double *matrix = (double *) malloc(matrix_size);
    double *output = (double *) malloc(matrix_size);



#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < _h * _r * _c; i++) {
        matrix[i] = (double) (rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int i = 0; i < heights * rows * cols; i++)
    {
        matrix[i] = (double)i;
    }
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


    // free space
    free(output);
    free(matrix);

    return 0;
}
