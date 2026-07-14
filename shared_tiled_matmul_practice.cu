#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


#define BLOCK_SIZE 16


void init_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i * cols + j] = (float)(rand() % 100);
}


__global__ void tiled_matmul_practice(float * A , float * B , float * C , int N , int K , int M){
    //few thing we always have is threadIdx , blockIdx , blockDim , threadDim (not sure if this one is used)

    //block idx represent globally of c what block we are making 

    int block_row = blockIdx.y; //what block row of A you need to access
    int block_col = blockIdx.x; //what block col of B you need to access 

    __shared__ float As[BLOCK_SIZE*4][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE*4];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x; //thread tells you the local idx inside that block we are computing of C which has in total BLOCK_SIZE*BLOCK_SIZE threads 

    //now we need to move A,B and C to the right index in memory 
    A += block_row * BLOCK_SIZE * K * 4;
    B += block_col * BLOCK_SIZE * 4 ;
    C += block_row * BLOCK_SIZE * M * 4 + block_col * BLOCK_SIZE * 4;

    float sums[4][4] = {0.0f};
    for (int b_idx = 0; b_idx < K; b_idx += BLOCK_SIZE) {
        for (int r = 0; r < 4; r++) {
            As[local_row + r * BLOCK_SIZE][local_col] =
                A[(local_row + r * BLOCK_SIZE) * K + local_col];
            Bs[local_row][local_col + r * BLOCK_SIZE] =
                B[local_row * M + local_col + r * BLOCK_SIZE];
            //each thread writes 4 values example ...
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            float a[4];
            float b[4];

            for (int r = 0; r < 4; r++) {
                a[r] = As[local_row + r * BLOCK_SIZE][i];
            }

            for (int c = 0; c < 4; c++) {
                b[c] = Bs[i][local_col + c * BLOCK_SIZE];
            }

            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    sums[r][c] += a[r] * b[c];
                }
            }
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * M;
    }

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            C[(local_row + r * BLOCK_SIZE) * M +
                (local_col + c * BLOCK_SIZE)] = sums[r][c];
        }
    }

}



int main(){

    int N = 4096;
    int K = 4096;
    int M = 4096;

    /*set up dummy A B MATIX INIT SOME VALUES */
    float * A = (float *) (malloc(N*K*sizeof(float)));
    float * B = (float *) (malloc(K*M*sizeof(float)));
    float * C = (float *) (malloc(N*M*sizeof(float)));

    init_random_matrix(A, N, K);
    init_random_matrix(B, K, M);

    float * d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*K*sizeof(float));
    cudaMalloc(&d_B, K*M*sizeof(float));
    cudaMalloc(&d_C, N*M*sizeof(float));

    cudaMemcpy(d_A, A, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*M*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridSize(
    (M + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4),
    (N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4)
        );

    tiled_matmul_practice<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, K, M);

    // measure floating point operations
    double flops = 2.0*N*K*M;

    // time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_matmul_practice<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, K, M);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", (flops / (milliseconds / 1000.0))/1e9);

    cudaMemcpy(C,d_C, N*M*sizeof(float), cudaMemcpyDeviceToHost);



    return 0;
}
