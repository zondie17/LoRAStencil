#include <iostream>
#include <omp.h>
#include "1d_utils.h"
#include <cmath>

const char *ShapeStr[4] = {
        "1d1r",
        "1d2r"
};

#define FILL_RANDOM
// #define FILL_INDEX

int HALO;


void printHelp() {
    const char *helpMessage =
            "Program name: lorastencil_1d\n"
            "Usage: lorastencil_1d shape input_size time_size\n"
            "Shape: 1d1r or 1d2r\n";
    printf("%s\n", helpMessage);
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
    }
    catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e) {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }


    // [0.25, 0.5, 0.25] => [0.015625, 0.09375, 0.234375, 0.3125, 0.234375, 0.09375, 0.015625]
    // [0.1, 0.2, 0.4, 0.2, 0,1] => [0.0111111, 0.0444444, 0.1333333, 0.2222222, 0.2888889, 0.2222222, 0.1333333, 0.0444444, 0.0111111]

    double param_1d1r[7] = {0.015625, 0.09375, 0.234375, 0.3125, 0.234375, 0.09375, 0.015625};
    double param_1d2r[9] = {0.0111111, 0.0444444, 0.1333333, 0.2222222, 0.2888889, 0.2222222, 0.1333333, 0.0444444, 0.0111111};


    double *param;

    switch (compute_shape) {
        case star_1d1r:
            param = param_1d1r;
            HALO = 3;
            break;
        case star_1d2r:
            param = param_1d2r;
            HALO = 4;
            break;
    }

    // print brief info

    printf("INFO: shape = %s, n = %d, times = %d\n", ShapeStr[compute_shape], n, times);

    int cols = n + 2 * HALO * 8;

    size_t input_size = (unsigned long) cols * sizeof(double);

    // allocate space

    double *input = (double *) malloc(input_size + sizeof(double));
    double *output = (double *) malloc(input_size + sizeof(double));

    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < cols + 1; i++) {
        input[i] = (double) (rand() % 10000);
    }
#elif defined(FILL_INDEX)
    if(compute_shape==star_1d1r_step3){
        for (int i = 0; i < cols + 1; i++)
        {
            if (i < HALO + 1 || i > cols - HALO)
                input[i] = 0;
            else
            {
                input[i] = i + 1 - (HALO+1);
                // printf("%d %lf\n",i,input[i]);
            }
        }
    }
    else{
        for (int i = 0; i < cols ; i++)
        {
            if (i < HALO  || i > cols - HALO -1)
                input[i] = 0;
            else
            {
                input[i] = i + 1 - HALO;
            }
            // printf("%d %lf\n",i,input[i]);
        }

    }
#endif

    switch (compute_shape) {
        case star_1d1r:
            gpu_1d1r(input, output, param, times, n);
            break;
        case star_1d2r:
            gpu_1d2r(input, output, param, times, n);
            break;
    }

    // free space
    free(output);
    free(input);

    return 0;
}