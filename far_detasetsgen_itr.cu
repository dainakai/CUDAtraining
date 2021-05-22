/*****************************************************************************
* Dai NAKAI
* Three dimensional Droplets distribution and its hologram generating software
* 2021/5/22
* 
******************************************************************************/
#include <bits/stdc++.h>
#include <cstdlib>
#include <sys/stat.h>
#include <time.h>
#include <cufft.h>
using namespace std;

#define WAVE_LEN 0.6328f
#define PI 3.14159265f
#define DX 10.0f

//CUDA function error chech macro
#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if(error != cudaSuccess){                                                   \
        printf("Error: %s:%d, ",__FILE__, __LINE__);                            \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \
}

//Confirm ref image when you change width or height
const char* ref_img_path = "./1024.bmp";
const char* output_dir = "./holograms/far_holo";
char output_dir_num[100];
unsigned char header_buf[1078];
char output_path[100];

const int height = 1024;
const int width = 1024;
const int depth = 2048;
const int particle_num = 10;
const int image_count = 1000;

//All units provided by micro meter
const float mean_diam = 50.0;
const float sd_diam = 10.0;
const float dist_to_cam = 50000.0;

const float peak_bright = 127;

void particle_diam_dist (float *diam, int index);
void particle_posi_dist (float *x, float *y, float*z);
void z_axis_sort (float *array, int left, int right);

__global__ void initialize_holo_plane (cufftComplex *holo);
__global__ void particle_volume (float *info, unsigned char *out);
__global__ void trans_func(cufftComplex *trans, float dist);
__global__ void extract_plane_from_vol(unsigned char* V, cufftComplex *plane, int num);
__global__ void fftshift_2D(cufftComplex *data);
__global__ void plane_complex_multiple (cufftComplex *A, cufftComplex *B, cufftComplex *C);
__global__ void holo_to_float_image (cufftComplex *data, float *image);
__global__ void two_dim_divide_for_fft (cufftComplex *data);

FILE *fp;
/**********************************main***********************************/
int main(int argc, char** argv){
    printf("%s Starting...\n", argv[0]);

    fp = fopen(ref_img_path,"rb");
    if(fp == NULL){
        printf("NO REFERENCE IMAGE! quitting...\n");
        exit(1);
    }
    int read_conf;
    read_conf = fread(header_buf, sizeof(unsigned char), 1078, fp);
    fclose(fp);

    float host_diam[particle_num];
    float host_posi_x[particle_num], host_posi_y[particle_num], host_posi_z[particle_num];
    unsigned char *dev_V;
    float *host_particle_info, *dev_particle_info;
    float *dev_float_image, float_image[height][width];
    unsigned char image_out[height][width];
    float dist_to_next_holo;

    int dev = 0;
    cudaSetDevice(dev);

    dim3 grid(particle_num, width, height), block(1);
    dim3 grid2(width,height), block2(1);

    CHECK(cudaMalloc((void **)&dev_particle_info, sizeof(float)*particle_num*4));
    CHECK(cudaMalloc((void **)&dev_V, sizeof(unsigned char)*width*height*particle_num));
    CHECK(cudaMalloc((void **)&dev_float_image, sizeof(float)*width*height));

    cufftHandle plan;
    cufftPlan2d(&plan,width,height,CUFFT_C2C);
    cufftComplex *devc_object, *devc_hologram, *devc_trans;
    CHECK(cudaMalloc((void **)&devc_object, sizeof(cufftComplex)*width*height)); 
    CHECK(cudaMalloc((void **)&devc_hologram, sizeof(cufftComplex)*width*height)); 
    CHECK(cudaMalloc((void **)&devc_trans, sizeof(cufftComplex)*width*height));

    sprintf(output_dir_num,"%s/num_%05d/",output_dir,particle_num);
    printf("%s\n",output_dir_num);
    mkdir(output_dir_num, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    srand((unsigned int)time(NULL));

    for(int count = 0; count < image_count; count++){
        particle_diam_dist(host_diam, particle_num);
        particle_posi_dist(host_posi_x, host_posi_y, host_posi_z);
        z_axis_sort(host_posi_z, 0, particle_num-1);

        host_particle_info = (float *)malloc(sizeof(float)*particle_num*4);
        for (int i = 0; i < particle_num; i++){
            host_particle_info[4*i] = host_diam[i];
            host_particle_info[4*i+1] = host_posi_x[i];
            host_particle_info[4*i+2] = host_posi_y[i];
            host_particle_info[4*i+3] = host_posi_z[i];
            // printf("%lf %lf %lf %lf\n",host_diam[i],host_posi_x[i],host_posi_y[i],host_posi_z[i]);
        }

        CHECK(cudaMemcpy(dev_particle_info, host_particle_info, sizeof(float)*particle_num*4, cudaMemcpyHostToDevice));

        particle_volume<<<grid, block>>>(dev_particle_info, dev_V);

        extract_plane_from_vol<<<grid2, block2>>>(dev_V, devc_object, 0);
        for (int itr = 0; itr < particle_num - 1; itr++) {
            cufftExecC2C(plan, devc_object, devc_object, CUFFT_FORWARD);
            fftshift_2D<<<grid2, block2>>>(devc_object);
            dist_to_next_holo = host_particle_info[4*(itr+1) + 3] - host_particle_info[4*itr + 3];
            trans_func<<<grid2, block2>>>(devc_trans, dist_to_next_holo);
            plane_complex_multiple<<<grid2, block2>>>(devc_object, devc_trans, devc_hologram);
            fftshift_2D<<<grid2, block2>>>(devc_hologram);
            cufftExecC2C(plan, devc_hologram, devc_hologram, CUFFT_INVERSE);
            two_dim_divide_for_fft<<<grid2,block2>>>(devc_hologram);
            extract_plane_from_vol<<<grid2, block2>>>(dev_V, devc_object, itr+1);
            plane_complex_multiple<<<grid2, block2>>>(devc_object, devc_hologram, devc_object);
        }
        cufftExecC2C(plan, devc_object, devc_object, CUFFT_FORWARD);
        fftshift_2D<<<grid2, block2>>>(devc_object);
        trans_func<<<grid2, block2>>>(devc_trans,  dist_to_cam + (float)depth*DX - host_particle_info[4*(particle_num-1) + 3]);
        plane_complex_multiple<<<grid2, block2>>>(devc_object, devc_trans, devc_hologram);
        fftshift_2D<<<grid2, block2>>>(devc_hologram);
        cufftExecC2C(plan, devc_hologram, devc_hologram, CUFFT_INVERSE);
        two_dim_divide_for_fft<<<grid2,block2>>>(devc_hologram);

        holo_to_float_image<<<grid2, block2>>>(devc_hologram, dev_float_image);

        CHECK(cudaMemcpy(float_image, dev_float_image, sizeof(float)*width*height, cudaMemcpyDeviceToHost));

        for (int y=0; y < height; y++) {
            for (int x=0; x < width; x++) {
                image_out[y][x] = (unsigned char)(peak_bright*float_image[y][x]);
            }
        }

        sprintf(output_path, "%s/%05d.bmp",output_dir_num,count);
        fp = fopen(output_path, "wb");
        fwrite(header_buf, sizeof(unsigned char), 1078, fp);
        fwrite(image_out, sizeof(unsigned char), width*height, fp);
        fclose(fp);

        printf("\n\n");
    }

    free(host_particle_info);
    cudaFree(dev_particle_info);
    cudaFree(dev_V);
    cufftDestroy(plan);
    cudaFree(devc_object);
    cudaFree(devc_hologram);
    cudaFree(devc_trans);
    cudaFree(dev_float_image);
    cudaDeviceReset();

    return 0;
}

void particle_posi_dist (float *x, float *y, float*z){
    for (int i = 0; i < particle_num; i++) {
        x[i] = rand() % width * DX;
        y[i] = rand() % height * DX;
        z[i] = rand() % depth * DX;
    }
}

void particle_diam_dist (float *diam, int index){
    float tmp1, tmp2, tmp3;
    for (int i = 0; i < index ; i++) {
        while(1){
            tmp1 = (float)rand() / (float)RAND_MAX;
            tmp2 = (float)rand() / (float)RAND_MAX;
            tmp3 = sqrt(-2.0*log(tmp1))*cos(2.0*M_PI*tmp2);
            diam[i] = sd_diam*tmp3 + mean_diam;
            if (diam[i] > 0.0) break;
        }
    }
}

void z_axis_sort (float *array, int left, int right){
    int tmp;
    int i = left;
    int j = right;
    float pivot = array[(i+j)/2];
    while(1){
        while(array[i] < pivot)
            i++;
        
        while(array[j] > pivot)
            j--;
        
        if(i >= j)
            break;

        tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;

        i++;
        j--;
    }
        if(i-1>left)
            z_axis_sort(array,left,i-1);

        if(j+1<right)
            z_axis_sort(array,j+1,right);
}

__global__ void particle_volume (float *info, unsigned char *out){
    int x, y, idx;
    x = blockIdx.y;
    y = blockIdx.z;
    idx = blockIdx.x;

    if( ((float)x*DX-info[4*idx + 1])*((float)x*DX-info[4*idx + 1]) + ((float)y*DX-info[4*idx + 2])*((float)y*DX-info[4*idx + 2]) > info[4*idx]*info[4*idx]/4.0 ){
        out[x + y*width + idx*width*height] = (unsigned char)1;
    }else{
        out[x + y*width + idx*width*height] = (unsigned char)0;
    }
}

__global__  void trans_func(cufftComplex *trans, float dist){
    int x, y;
    x = blockIdx.x;
    y = blockIdx.y;

    float c0,c1,c2, tmp;

    c0 = 2.0*PI*dist/WAVE_LEN;
    c1 = WAVE_LEN*WAVE_LEN/width/width/DX/DX;
    c2 = WAVE_LEN*WAVE_LEN/height/height/DX/DX;
    
    tmp = c0*sqrt(1.0-c1*((float)x-(float)width/2.0)*((float)x-(float)width/2.0)-c2*((float)y-(float)height/2.0)*((float)y-(float)height/2.0));
    trans[x + y*width].x = cos(tmp);
    trans[x + y*width].y = sin(tmp);
}

__global__ void extract_plane_from_vol(unsigned char* V, cufftComplex *plane, int num){
    int x,y;
    x = blockIdx.x;
    y = blockIdx.y;

    plane[x + width*y].x = (float)V[x + width*y + width*height*num];
    plane[x + width*y].y = 0.0;
}

__global__ void plane_complex_multiple (cufftComplex *A, cufftComplex *B, cufftComplex *C){
	int x = blockIdx.x;
    int y = blockIdx.y;

    float tmp1, tmp2;

    tmp1 = A[x + width*y].x * B[x + width*y].x - A[x + width*y].y * B[x + width*y].y;
    tmp2 = A[x + width*y].x * B[x + width*y].y + A[x + width*y].y * B[x + width*y].x;

    C[x + width*y].x = tmp1;
    C[x + width*y].y = tmp2;
}

__global__ void fftshift_2D(cufftComplex *data){
	int x = blockIdx.x;
    int y = blockIdx.y;
    cufftComplex temp1,temp2;
    
    if((x < width/2) && (y < height/2)){
        temp1 = data[x + width*y];
        data[x + width*y] = data[x + width/2 + width*(y + height/2)];
        data[x + width/2 + width*(y + height/2)] = temp1;
    }
    if((x < width/2) && (y >= height/2)){
        temp2 = data[x + width*y];
        data[x + width*y] = data[x + width/2 + width*(y - height/2)];
        data[x + width/2 + width*(y - height/2)] = temp2;
    }
}

__global__ void holo_to_float_image (cufftComplex *data, float *image){
	int x = blockIdx.x;
    int y = blockIdx.y;

    image[x + width*y] = sqrt(data[x + width*y].x*data[x + width*y].x + data[x + width*y].y*data[x + width*y].y);

}

__global__ void two_dim_divide_for_fft (cufftComplex *data){
    int x = blockIdx.x;
    int y = blockIdx.y;

    data[x + width*y].x /= width*height;
    data[x + width*y].y /= width*height;

} 