#include <bits/stdc++.h>
using namespace std;

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if(1){                                                                      \
        printf("Error: %s:%d, ",__FILE__, __LINE__);                            \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
                                                                        \
    }                                                                           \
}

const float a = 1.0;

__global__ void test(void){
    int x,y,z;
    x = blockIdx.x;
    y = blockIdx.y;
    z = blockIdx.z;
    printf("%d %d %d %lf\n",x,y,z,a);
}

int main(){
    dim3 grid(2,3,4), block(1);
    test<<<grid,block>>>();

    cudaDeviceReset();
}

