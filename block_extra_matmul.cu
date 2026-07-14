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
    
    __shared__ float As[BLOCK_SIZE*2][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE*2];

    // blockDim , blockIdx , threadIdx 
    // blockIdx tells us the global index of the blocks we wanna be in 
    // threadIdx tells us the local index inside that block we are in 
    // since we are estimating a BLOCK_SIZE*BLOCK_SIZE size of C matrix 

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    //move our pointers 
    A += blockRow * BLOCK_SIZE * K; //stride to start of the correct row block 
    B += blockCol * BLOCK_SIZE*2; //B always starts at row 0 but diff col block 
    C += blockRow * BLOCK_SIZE * 2  * M + blockCol * BLOCK_SIZE*2;

    float sum = 0.0f;
    float sum2= 0.0f;
    float sum3= 0.0f;
    float sum4 = 0.0f;

    for (int b_idx=0;b_idx<K;b_idx+=BLOCK_SIZE){

        As[localRow][localCol] = A[localRow*K+localCol];
        Bs[localRow][localCol] = B[localRow*M+localCol];
        Bs[localRow][localCol+BLOCK_SIZE] = B[localRow*M+localCol+BLOCK_SIZE];
        As[localRow+BLOCK_SIZE][localCol] = A[(localRow+BLOCK_SIZE)*K+localCol];



        __syncthreads(); //so every 2d thread now writes and waits for all to write the BLOCK_SIZE*BLOCK_SIZE smem fill it 
 
        A+= BLOCK_SIZE;  // A is easy to move since its memory flat 1d way we just advance blocksize to move pointer to start of next block of A
        B+= BLOCK_SIZE * M; // Be is hard since we hace to advance a row in B so we need to move and stride 

        for (int i = 0 ; i<BLOCK_SIZE;i++){
            //this loop moves locally inside the BLOCK_SIZE*BLOCK_SIZE shared mem 
            sum+= As[localRow][i] * Bs[i][localCol];
            sum2+= As[localRow][i] * Bs[i][localCol+BLOCK_SIZE];
            sum3+= As[localRow+BLOCK_SIZE][i] * Bs[i][localCol];
            sum4+= As[localRow+BLOCK_SIZE][i] * Bs[i][localCol+BLOCK_SIZE];
        }
        __syncthreads(); //this makes all the threads locally in that smem to complete 

        //now we advance to next block size until all K is covered
    }

    C[localRow*M+localCol] = sum;
    C[localRow*M+localCol+BLOCK_SIZE]= sum2;
    C[(localRow+BLOCK_SIZE)*M+localCol] = sum3;
    C[(localRow+BLOCK_SIZE)*M+localCol+BLOCK_SIZE] = sum4;

    //this writes one element in that block_size*block_size C and all threads 16*16 write themselves


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