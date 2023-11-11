#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define SIZE 1024
#define num_threads 8

int a[SIZE][SIZE];
int b[SIZE][SIZE];
long int c[SIZE][SIZE];

void read();
void* slave(void *arg);

void read() {
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            a[i][j] = (i + 1) / 2;
            b[i][j] = (i + 1) / 2;
            c[i][j] = 0;
        }
    }
}

void* slave(void *arg) {
    int i, j, k, id;
    id = (intptr_t)arg;
    int start, end, specific_point;
    specific_point = SIZE / num_threads;
    start = id * specific_point;
    end = start + specific_point;
    for (i = start; i < end; i++) {
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return NULL;
}

int main() {
    int i;
    pthread_t thread[num_threads];
    double start_time, end_time;
    read();
    start_time = time(NULL);
    for (i = 0; i < num_threads; i++) {
        if (pthread_create(&thread[i], NULL, slave, (void *)(intptr_t)i) != 0) {
            perror("Pthread_create fails");
            exit(1);
        }
    }
    for (i = 0; i < num_threads; i++) {
        if (pthread_join(thread[i], NULL) != 0) {
            perror("Pthread_join fails");
            exit(1);
        }
    }
    end_time = time(NULL);
    printf("PTHREAD CODE TIME TOOK %lf\n", difftime(end_time, start_time));
    return 0;
}
