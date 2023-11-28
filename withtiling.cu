
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMultiplication(float* a, float* b, float* c, int M, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    float sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < N) {
            tileA[ty][tx] = a[row * N + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0;
        }

        if (col < N && t * TILE_SIZE + ty < N) {
            tileB[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

int main() {
    int M = 1000;
    int N = 500;
    float* a, * b, * c;
    float* d_a, * d_b, * d_c;
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

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    

    printf("Execution time: %.2f ms\n", milliseconds);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

