#include <stdio.h>
#include <sys/time.h>

void matrixMultiplication(float *a, float *b, float *c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

double getElapsedTime(struct timeval start, struct timeval stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    int M = 1000;
    int N = 500;
    float *a, *b, *c;

    int size_a = M * N * sizeof(float);
    int size_b = N * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    a = (float*)malloc(size_a);
    b = (float*)malloc(size_b);
    c = (float*)malloc(size_c);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = i - j;
        }
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);
    matrixMultiplication(a, b, c, M, N);
    gettimeofday(&stop, NULL);

   

    double elapsedTime = getElapsedTime(start, stop);
    printf("Execution time: %.2f ms\n", elapsedTime);

    free(a);
    free(b);
    free(c);

    return 0;
}
