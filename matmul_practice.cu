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



__global__ void naive_matmul_tiled(float * A, float * B , float * C ,int N , int K , int M){
    // this basically widens b to be BLOCK_SIZE, BLOCK_SIZE *2 so for each block load of A we do 2 blocks of B which helps us do 2 blocks of C estimated 
    // so more reuse of A 

    __shared__ float As[BLOCK_SIZE*2][BLOCK_SIZE*2];
    __shared__ float Bs[BLOCK_SIZE*2][BLOCK_SIZE*2];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int block_idx_row = blockIdx.y;
    int block_idx_col = blockIdx.x;

    //move pointers for A , B and C to the correct locations 
    A+= block_idx_row * BLOCK_SIZE * K;
    B+= block_idx_col*BLOCK_SIZE;
    C+= block_idx_row*BLOCK_SIZE * M + BLOCK_SIZE * block_idx_col;


    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (int b_idx=0;b_idx<K;b_idx+=BLOCK_SIZE*2){

        int flat_tid = threadIdx.y  * BLOCK_SIZE + threadIdx.x;

        int vecs_per_row = (BLOCK_SIZE*2) / 4;
        int shared_row = flat_tid / vecs_per_row; //this tells use the row we are in for that loading isnce each thread loads 4 elements 
        //this tells us exactly locally what row are we in that 32*32 block becuase if each is loacing 4 you would have 8 total chunks to load over threads
        int shared_col = (flat_tid%vecs_per_row) *4; //locally tells us what chunk of the 8 sized row vector you are loading and you move by 4 because each is loading 4 so offset 

        int a_global_row = threadIdx.y * BLOCK_SIZE * 2 + shared_row;
        int a_global_col = shared_col + b_idx; 

        int b_global_row = b_idx + shared_row;
        int b_global_col = blockIdx.x * BLOCK_SIZE*2 + shared_col;

        float4 a_vec = reinterpret_cast<float4*>(&A[a_global_row*K+a_global_col])[0];
        float4 b_vec = reinterpret_cast<float4*>(&B[b_global_row*M+b_global_col])[0];

        reinterpret_cast<float4*>(&As[shared_row][shared_col])[0] = a_vec;
        reinterpret_cast<float4*>(&Bs[shared_row][shared_col])[0] = b_vec;

        __syncthreads();

        for (int k=0;k<BLOCK_SIZE*2;k++){
            float a0 = As[threadIdx.y][k];
            float a1 = As[threadIdx.y+BLOCK_SIZE][k];

            float b0 = Bs[k][threadIdx.x];
            float b1 = Bs[k][threadIdx.x+BLOCK_SIZE];

            acc00+= a0*b0;
            acc01+= a0*b1;
            acc10+= a1*b0;
            acc11+= a1*b1;



        }

        __syncthreads();


    }

    int c_global_row0 = blockIdx.y * BLOCK_SIZE * 2 + threadIdx.y;
    int c_global_row1 = c_global_row0 + BLOCK_SIZE;

    int c_global_col0 = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    int c_global_col1 = c_global_col0 + BLOCK_SIZE;

    C[c_global_row0 * M + c_global_col0] = acc00;
    C[c_global_row0 * M + c_global_col1] = acc01;
    C[c_global_row1 * M + c_global_col0] = acc10;
    C[c_global_row1 * M + c_global_col1] = acc11;




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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
    (M + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2),
    (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)
        );

    // Launch the kernel
    naive_matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, K, M);

    // measure floating point operations
    double flops = 2.0*N*K*M;

    // time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, K, M);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);
    printf("TFLOPS: %f\n", (flops / (milliseconds / 1000.0))/1e12);

    cudaMemcpy(C,d_C, N*M*sizeof(float), cudaMemcpyDeviceToHost);



    return 0;
}