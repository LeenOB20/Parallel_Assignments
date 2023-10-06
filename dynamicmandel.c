#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define IMG_WIDTH 800
#define IMG_HEIGHT 800
#define MAX_ITERATIONS 1000

// Calculate the Mandelbrot set for a given point
int mandelbrot(double x, double y) {
    double real = 0.0, imag = 0.0;
    int iterations = 0;
    while (real * real + imag * imag <= 4.0 && iterations < MAX_ITERATIONS) {
        double temp = real * real - imag * imag + x;
        imag = 2.0 * real * imag + y;
        real = temp;
        iterations++;
    }
    return iterations;
}

int main(int argc, char** argv) {

    double start_time, end_time, elapsed_time;
    int proc_rank, proc_size;
    
    start_time = MPI_Wtime();
    double start_parallel_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

    // Calculate the portion of the Mandelbrot set that this process is responsible for
    int rows_per_proc = IMG_HEIGHT / proc_size;
    int start_row = proc_rank * rows_per_proc;
    int end_row = (proc_rank + 1) * rows_per_proc;
    if (proc_rank == proc_size - 1) {
        end_row = IMG_HEIGHT;
    }

    int *image = (int*)malloc(IMG_WIDTH * (end_row - start_row) * sizeof(int));
    
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < IMG_WIDTH; i++) {
            double x = (i - IMG_WIDTH/2.0) * 4.0 / IMG_WIDTH;
            double y = (j - IMG_HEIGHT/2.0) * 4.0 / IMG_WIDTH;
            image[(j - start_row) * IMG_WIDTH + i] = mandelbrot(x, y);
        }
    }
    
    // Start parallel execution time measurement
    start_parallel_time = MPI_Wtime();

    // Combine the results from all processes
    int *result = NULL;
    if (proc_rank == 0) {
        result = (int*)malloc(IMG_WIDTH * IMG_HEIGHT * sizeof(int));
    }

    MPI_Gather(image, IMG_WIDTH * rows_per_proc, MPI_INT, result, IMG_WIDTH * rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // End parallel execution time measurement
    double end_parallel_time = MPI_Wtime();
    double parallel_time = end_parallel_time - start_parallel_time;

    // Reduce the parallel time across all processes
    double total_parallel_time;
    MPI_Reduce(&parallel_time, &total_parallel_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
        // Calculate and print timing information
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        double sequential_time = elapsed_time - total_parallel_time;
        printf("Sequential program took %f seconds\n", sequential_time);
        printf("Parallel program took %f seconds\n", total_parallel_time);

        // Write the Mandelbrot set to a file
        FILE *fp = fopen("mandelbrot.ppm", "wb");
        fprintf(fp, "P6\n%d %d\n255\n", IMG_WIDTH, IMG_HEIGHT);
        for (int j = 0; j < IMG_HEIGHT; j++) {
            for (int i = 0; i < IMG_WIDTH; i++) {
                int index = j * IMG_WIDTH + i;
                unsigned char r, g, b;
                if (result[index] == MAX_ITERATIONS) {
                    r = g = b = 0;
                } else {
                    r = (result[index] % 256);
                    g = (result[index] % 256);
                    b = (result[index] % 256);
                }
                fwrite(&r, 1, 1, fp);
                fwrite(&g, 1, 1, fp);
                fwrite(&b, 1, 1, fp);
            }
        }
        fclose(fp);
        free(result);
    }

    free(image);
    MPI_Finalize();

    return 0;
}
