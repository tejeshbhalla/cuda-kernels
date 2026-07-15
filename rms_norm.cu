#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>





void __random_bf16_vector(__nv_bfloat16 *vector, int size) {
    for (int i = 0; i < size; i++) {
        float value = (float)(rand() % 100) / 100.0f;
        vector[i] = __float2bfloat16(value);
    }
}


__device__ float warp_reduce_sum(float val){
    for (int offset=16;offset>0;offset=offset/2){
        val+= __shfl_down_sync(0xffffffff, val, offset);

    }

    return val;

}

__global__ void rms_norm_kernel(__nv_bfloat16 *input, __nv_bfloat16 *weight, __nv_bfloat16 *output, int D,float eps) {
    //left for user to write the kernel for rms_norm

    __shared__ float smem[8]; //8 because we have 8 warps so 8 slots of memory 

    int tid = threadIdx.x; // this tells us the thread we are in we have 256 threads 
    int lane_id = tid % 32; // lane is a local thread in a warp cycles from 0-32
    int warp_id = tid/32; // warp id is /32 it goes from 0-8

    float squared_sum = 0.0f;

    for (int i=tid;i<D;i+=blockDim.x){
        float val = __bfloat162float(input[i]);
        squared_sum+= val * val;
    } //uptill here inside each warp 32 threads individually have a squared sum 


    squared_sum = warp_reduce_sum(squared_sum); //in each warp 32 threads reduce sum 

    if (lane_id==0){
        smem[warp_id] = squared_sum; // for each of 8 warps we right within their reduced 32 sum squared here 
    }

    __syncthreads();

    //we have 8 warps each writing to smem sum now we need to reduce within them 

    if (warp_id==0){
        if (lane_id<blockDim.x/32){
            squared_sum = smem[lane_id];
        } else{
            squared_sum = 0.0f;
        }

        squared_sum = warp_reduce_sum(squared_sum);

        if (lane_id==0){
            smem[0] = squared_sum;
        }
    }

    __syncthreads();

    float rstd  = rsqrtf(smem[0]/(float) D + eps);

    for (int i=tid;i<D;i+=blockDim.x){
        float val = __bfloat162float(input[i]);
        float weight_val = __bfloat162float(weight[i]);
        float output_val = val * rstd * weight_val;
        output[i] = __float2bfloat16(output_val);
    }






}


void print_vector(__nv_bfloat16 *vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", __bfloat162float(vector[i]));
    }
    printf("\n");
}


int main(){
    int D = 4096; //no of dimensions / features 

    __nv_bfloat16 * input = (__nv_bfloat16 * ) malloc(D*sizeof(__nv_bfloat16));
    __nv_bfloat16 * output = (__nv_bfloat16 * ) malloc(D*sizeof(__nv_bfloat16));
    __nv_bfloat16 * weight = (__nv_bfloat16 * ) malloc(D*sizeof(__nv_bfloat16));

    __random_bf16_vector(input, D);
    __random_bf16_vector(weight, D);
    __random_bf16_vector(output, D);


    printf("Input vector:\n");
    print_vector(input, D);
    printf("Weight vector:\n");
    print_vector(weight, D);
    printf("Output vector:\n");
    print_vector(output, D);


    rms_norm_kernel<<<1,256>>>(input,weight,output,D,1e-5);

    //print the output vector after kernel execution
    printf("Output vector after RMSNorm:\n");
    print_vector(output, D);


    return 0;
}


