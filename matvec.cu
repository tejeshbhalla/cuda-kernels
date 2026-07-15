#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>







void init_random_bf16_matrix(__nv_bfloat16 *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float value = (float)(rand() % 100) / 100.0f;
        matrix[i] = __float2bfloat16(value); //converts a value from float to bfloat16
    }
}

void init_random_bf16_vector(__nv_bfloat16 *vector, int size) {
    for (int i = 0; i < size; i++) {
        float value = (float)(rand() % 100) / 100.0f;
        vector[i] = __float2bfloat16(value);
    }
}



__device__ float warp_reduce_sum(float val){
    for (int offset=16;offset>0;offset=offset/2){
        val+= __shfl_down_sync(0xffffffff,val,offset);
    }

    return val;
}


__global__ void matvec_kernel(
    __nv_bfloat16 *A,  //matrix weight
    __nv_bfloat16 *x,  //vector x 
    float *y,          //output y 
    int M , int K
){

    int warps_per_block = blockDim.x / 32;  //total threads/32 
    int warp_id = threadIdx.x/32;  //what row data we are processing 
    int lane_id = threadIdx.x % 32; // in each row warps do cycles of 32 threads so this tells local index of thread so say warp 0 thread 31 does 31%32 thats 31 0-31 cycle

    int row = warps_per_block * blockIdx.x + warp_id;

    if (row>=M) return;

    __nv_bfloat16 * row_ptr = A + (row * K);
    float sum = 0.0f;
    
    int k = lane_id  * 4; //each local thread does 4 elements so we start it at 4 for 0 its 0 for 1 its 4 

    for (;k<=K-128;k+=128){
        uint2 packed = __ldg(reinterpret_cast<uint2*>(row_ptr+k));
        __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed); //this gives us all the 4 values 
        
        sum+= __bfloat162float(vals[0]) * __bfloat162float(__ldg(x + k));
        sum+= __bfloat162float(vals[1]) * __bfloat162float(__ldg(x + k + 1));
        sum+= __bfloat162float(vals[2]) * __bfloat162float(__ldg(x + k + 2));
        sum+= __bfloat162float(vals[3]) * __bfloat162float(__ldg(x + k + 3));

    }

    for (;k<K;k+=128){
        if (k<K) sum+= __bfloat162float(__ldg(row_ptr + k)) * __bfloat162float(__ldg(x + k));
        if (k+1<K) sum+= __bfloat162float(__ldg(row_ptr + k+1)) * __bfloat162float(__ldg(x + k + 1));
        if (k+2<K) sum+= __bfloat162float(__ldg(row_ptr + k+2)) * __bfloat162float(__ldg(x + k + 2));
        if (k+3<K) sum+= __bfloat162float(__ldg(row_ptr + k+3)) * __bfloat162float(__ldg(x + k + 3));
    }

    // each lane has processed a local parition of the sum accorss 32 threads/ lanes we need to warp reduce too 
    
    sum = warp_reduce_sum(sum);

    if (lane_id==0){
        y[row] = sum;
    }


}


void print_output(float *y, int M) {
    for (int i = 0; i < M; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }
}




int main(){

    int M  = 4096;  //M rows
    int K = 4096;  //K cols 

    __nv_bfloat16 *A = (__nv_bfloat16 *)malloc(M*K*sizeof(__nv_bfloat16));
    __nv_bfloat16 *x = (__nv_bfloat16 *)malloc(K*sizeof(__nv_bfloat16));

    float *y = (float *)malloc(M*sizeof(float));

    init_random_bf16_matrix(A, M, K);
    init_random_bf16_vector(x, K);


    __nv_bfloat16 *d_A, *d_x;
    float *d_y;

    cudaMalloc(&d_A,M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_x,K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_y,M*sizeof(float));

    cudaMemcpy(d_A,A,M*K*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    //kernel launch stuff 

    int warps_per_block = 8; //8 warps per block each warp does one row 
    int threads = warps_per_block*32; //256 threads launching 
    int blocks  = (M+warps_per_block-1) / warps_per_block; //we have M rows and each warp handles 1 row , thus working out to ceil(M,warps_per_block)


    matvec_kernel<<<blocks,threads>>>(d_A,d_x,d_y,M,K);
    cudaMemcpy(y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    print_output(y, M);


    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    free(A);
    free(x);
    free(y);

    return 0;
}
