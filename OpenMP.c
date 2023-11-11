#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#define MAX_THREADS 4
int N, M, P;
double **A, **B, **C;
int main(int argc, char *argv[]) {
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    P = atoi(argv[3]);
    A = (double**) malloc(N * sizeof(double*));
    B = (double**) malloc(M * sizeof(double*));
    C = (double**) malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        A[i] = (double*) malloc(M * sizeof(double));
        C[i] = (double*) malloc(P * sizeof(double));
        for (int j = 0; j < M; j++) {
            A[i][j] = rand() % 10;
        }
    }
    for (int i = 0; i < M; i++) {
        B[i] = (double*) malloc(P * sizeof(double));
        for (int j = 0; j < P; j++) {
            B[i][j] = rand() % 10;
        }
    }
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0;
            for (int k = 0; k < M; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;
    double flops = 2 * N * M * P;
    double mflops = flops / (elapsed_time * 1e6);
    double speedup = mflops / (N * M * P / elapsed_time);
    double efficiency = speedup / MAX_THREADS;
    double scalability = speedup / (N * M * P / elapsed_time / MAX_THREADS);
    printf("Elapsed time (s): %f\n", elapsed_time);
    printf("Speedup: %f\n", speedup);
    printf("Efficiency: %f\n", efficiency);
    printf("Scalability: %f\n", scalability);
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(C[i]);
    }
    for (int i = 0; i < M; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
