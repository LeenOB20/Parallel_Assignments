#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#define MAX_THREADS 4
int N, M, P;
double **A, **B, **C;
pthread_t threads[MAX_THREADS];
pthread_mutex_t lock;
void *multiply(void *arg) {
    int tid = *(int*) arg;
    int start = tid * (N / MAX_THREADS);
    int end = (tid + 1) * (N / MAX_THREADS);
    for (int i = start; i < end; i++) {
        for (int j = 0; j < P; j++) {
            double sum = 0;
            for (int k = 0; k < M; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    pthread_exit(NULL);
}
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
    pthread_mutex_init(&lock, NULL);
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < MAX_THREADS; i++) {
        int *tid = (int*) malloc(sizeof(int));
        *tid = i;
        pthread_create(&threads[i], NULL, multiply, (void*) tid);
    }
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +(end_time.tv_usec - start_time.tv_usec) / 1e6;
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
    pthread_mutex_destroy(&lock);
    pthread_exit(NULL);
    return 0;
}
