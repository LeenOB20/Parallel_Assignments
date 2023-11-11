#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define num_threads 8

int main() {
    int i, j, k;
    double a[num_threads][num_threads], b[num_threads][num_threads], c[num_threads][num_threads];

    for (i = 0; i < num_threads; ++i) {
        for (j = 0; j < num_threads; ++j) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c[i][j] = 0;
        }
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel for private(i,j,k) shared(a,b,c) schedule(dynamic) 
    {
        for (i = 0; i < num_threads; ++i) {
            for (j = 0; j < num_threads; ++j) {
                for (k = 0; k < num_threads; ++k) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("openMP CODE TIME TOOK: %f seconds\n", end_time - start_time);

    return 0;
}
