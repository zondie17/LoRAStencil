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


// Check the correctness of the results or not
// #define CHECK_ERROR
const double tolerance = 1e-7;

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define ABS(x, y) (((x) > (y)) ? ((x) - (y)): ((y) - (x)))


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


void test_cpu(double *in, double * out, double *param, const int input_m, const int input_n) {
    for (int row = 4; row < input_m - 4; row++) {
        for (int col = 4; col < input_n - 4; col++) {
            out[IDX(row, col, input_n)] =
                param[0] * in[IDX(row - 3, col - 3, input_n)] +
                param[1] * in[IDX(row - 3, col - 2, input_n)] +
                param[2] * in[IDX(row - 3, col - 1, input_n)] +
                param[3] * in[IDX(row - 3, col, input_n)] +
                param[4] * in[IDX(row - 3, col + 1, input_n)] +
                param[5] * in[IDX(row - 3, col + 2, input_n)] +
                param[6] * in[IDX(row - 3, col + 3, input_n)] +
                param[7] * in[IDX(row - 2, col - 3, input_n)] +
                param[8] * in[IDX(row - 2, col - 2, input_n)] +
                param[9] * in[IDX(row - 2, col - 1, input_n)] +
                param[10] * in[IDX(row - 2, col, input_n)] +
                param[11] * in[IDX(row - 2, col + 1, input_n)] +
                param[12] * in[IDX(row - 2, col + 2, input_n)] +
                param[13] * in[IDX(row - 2, col + 3, input_n)] +
                param[14] * in[IDX(row - 1, col - 3, input_n)] +
                param[15] * in[IDX(row - 1, col - 2, input_n)] +
                param[16] * in[IDX(row - 1, col - 1, input_n)] +
                param[17] * in[IDX(row - 1, col, input_n)] +
                param[18] * in[IDX(row - 1, col + 1, input_n)] +
                param[19] * in[IDX(row - 1, col + 2, input_n)] +
                param[20] * in[IDX(row - 1, col + 3, input_n)] +
                param[21] * in[IDX(row, col - 3, input_n)] +
                param[22] * in[IDX(row, col - 2, input_n)] +
                param[23] * in[IDX(row, col - 1, input_n)] +
                param[24] * in[IDX(row, col, input_n)] +
                param[25] * in[IDX(row, col + 1, input_n)] +
                param[26] * in[IDX(row, col + 2, input_n)] +
                param[27] * in[IDX(row, col + 3, input_n)] +
                param[28] * in[IDX(row + 1, col - 3, input_n)] +
                param[29] * in[IDX(row + 1, col - 2, input_n)] +
                param[30] * in[IDX(row + 1, col - 1, input_n)] +
                param[31] * in[IDX(row + 1, col, input_n)] +
                param[32] * in[IDX(row + 1, col + 1, input_n)] +
                param[33] * in[IDX(row + 1, col + 2, input_n)] +
                param[34] * in[IDX(row + 1, col + 3, input_n)] +
                param[35] * in[IDX(row + 2, col - 3, input_n)] +
                param[36] * in[IDX(row + 2, col - 2, input_n)] +
                param[37] * in[IDX(row + 2, col - 1, input_n)] +
                param[38] * in[IDX(row + 2, col, input_n)] +
                param[39] * in[IDX(row + 2, col + 1, input_n)] +
                param[40] * in[IDX(row + 2, col + 2, input_n)] +
                param[41] * in[IDX(row + 2, col + 3, input_n)] +
                param[42] * in[IDX(row + 3, col - 3, input_n)] +
                param[43] * in[IDX(row + 3, col - 2, input_n)] +
                param[44] * in[IDX(row + 3, col - 1, input_n)] +
                param[45] * in[IDX(row + 3, col, input_n)] +
                param[46] * in[IDX(row + 3, col + 1, input_n)] +
                param[47] * in[IDX(row + 3, col + 2, input_n)] +
                param[48] * in[IDX(row + 3, col + 3, input_n)];
        }
    }
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

    bool breakdown = false;

    double param_box_2d1r[49] = {0.0};
    double param_star_2d3r[49] = {0.0};

    for (double & i : param_box_2d1r) {
        i = 0.021;
    }

    for (double &i: param_box_2d1r) {
        i = 0;
    }

    // box_2d param init
    int param_box_num = 1;
    for (int i = -3; i < 1; i++) {
        for (int j = -3; j < 1; j++) {
            if (i <= j) {
                param_box_2d1r[(i + 3) * 7 + (j + 3)] = param_box_num;
                param_box_2d1r[(-i + 3) * 7 + (j + 3)] = param_box_num;
                param_box_2d1r[(i + 3) * 7 + (-j + 3)] = param_box_num;
                param_box_2d1r[(-i + 3) * 7 + (-j + 3)] = param_box_num;
                param_box_2d1r[(j + 3) * 7 + (i + 3)] = param_box_num;
                param_box_2d1r[(-j + 3) * 7 + (i + 3)] = param_box_num;
                param_box_2d1r[(j + 3) * 7 + (-i + 3)] = param_box_num;
                param_box_2d1r[(-j + 3) * 7 + (-i + 3)] = param_box_num;
                param_box_num++;
            }
        }
    }
    param_box_2d1r[3*7 + 3] = 8;
    // [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0,
    //    2.0, 5.0, 6.0, 7.0, 6.0, 5.0, 2.0,
    //    3.0, 6.0, 8.0, 9.0, 8.0, 6.0, 3.0,
    //    4.0, 7.0, 9.0, 8.0, 9.0, 7.0, 4.0,
    //    3.0, 6.0, 8.0, 9.0, 8.0, 6.0, 3.0,
    //    2.0, 5.0, 6.0, 7.0, 6.0, 5.0, 2.0,
    //    1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]

    // star_2d3r param init
    int param_star_2d3r_num = 1;
    for (int i = -3; i < 1; i++) {
        param_star_2d3r[(i + 3) * 7 + 3] = param_star_2d3r_num;
        param_star_2d3r[(-i + 3) * 7 + 3] = param_star_2d3r_num;
        param_star_2d3r[3 * 7 + (i + 3)] = param_star_2d3r_num;
        param_star_2d3r[3 * 7 + (-i + 3)] = param_star_2d3r_num;
        param_star_2d3r_num++;
    }

    // star_2d1r param init
    double param_star_2d1r[49] = {
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 2, 4, 2, 0, 0,
        0, 2, 4, 8, 4, 2, 0,
        1, 4, 8, 16, 8, 4, 1,
        0, 2, 4, 8, 4, 2, 0,
        0, 0, 2, 4, 2, 0, 0,
        0, 0, 0, 1, 0, 0, 0
    };


    double *param;
    switch (compute_shape) {
        case box_2d1r:
            param = param_box_2d1r;
            break;
        case star_2d1r:
            param = param_star_2d1r;
            break;
        case star_2d3r:
            param = param_star_2d3r;
            break;
        case box_2d3r:
            param = param_box_2d1r;
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

    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 0.0;
    }
#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double) (rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int i = 4; i < rows - 4; i++)
    {
        for (int j = 4; j < cols - 4; j++)
        {
            matrix[i * cols + j] = (double)((i - 4) * (cols - 8) + j - 4);
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


#if defined(CHECK_ERROR)
    std::cout << arg1 << std::endl;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            std::cout << param[i * 7 + j] << " ";
        }
        std::cout << std::endl;
    }
#endif


    switch (compute_shape) {
        case star_2d1r:
            gpu_star_2d1r(matrix, output, param, times, m, n);
            break;
        case star_2d3r: {
            gpu_star_2d3r(matrix, output, param, times, m, n);
            break;
        }
        case box_2d1r:
        case box_2d3r:
            gpu_box_2d3r(matrix, output, param, times, m, n);
            break;
    }

#if defined(CHECK_ERROR)
    printf("\nChecking Correctness... \n");
    double *naive[2];
    naive[0] = (double *) malloc(matrix_size);
    naive[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < rows * cols; i++) {
        naive[0][i] = matrix[i];
        naive[1][i] = 0;
    }

    double *lora[2];
    lora[0] = (double *) malloc(matrix_size);
    lora[1] = (double *) malloc(matrix_size);

    for (int i = 0; i < rows * cols; i++) {
        lora[0][i] = matrix[i];
        lora[1][i] = 0;
    }

    test_cpu(naive[0], naive[1], param, rows, cols);
    switch (compute_shape) {
        case star_2d1r:
            gpu_star_2d1r(lora[0], lora[1], param, 1, m, n);
        break;
        case star_2d3r: {
            gpu_star_2d3r(lora[0], lora[1], param, 1, m, n);
            break;
        }
        case box_2d1r:
        case box_2d3r:
            gpu_box_2d3r(lora[0], lora[1], param, 1, m, n);
        break;
    }


    printf("Comparing naive and lora\n");
    for (int row = 4; row < rows - 4; row++) {
        for (int col = 4; col < cols - 4; col++) {
            if (ABS(naive[1][IDX(row, col, cols)], lora[1][IDX(row, col, cols)]) > 1e-7) {
                printf("row = %d, col = %d, naive = %lf, lora = %lf\n", row, col, naive[1][IDX(row, col, cols)], lora[1][IDX(row, col, cols)]);
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