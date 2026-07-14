#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>



void init_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i * cols + j] = (float)(rand() % 100);
}



__global__ void naive_matmul(float * A, float * B , float * C ,int N , int K , int M){

    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<N && col < M){
        float sum = 0.0f;
        for (int l=0;l<K;l++){

            sum+= A[row*K+l] * B[l*M+col];

        }

        C[row*M+col] = sum;
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

    dim3 blockSize(16,16);
    dim3 gridSize((M+blockSize.x-1)/blockSize.x,(N+blockSize.y-1)/blockSize.y);

    
    

    // measure floating point operations
    double flops = 2.0*N*K*M;

    // time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_matmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, K, M);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", (flops / (milliseconds / 1000.0))/1e9);

    cudaMemcpy(C,d_C, N*M*sizeof(float), cudaMemcpyDeviceToHost);



    return 0;
}