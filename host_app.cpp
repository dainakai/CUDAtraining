#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

#define WAVE_LEN 0.6328f
#define PI 3.14159265f
#define DX 10.0f

//Confirm ref image when you change width or height. 
const char* ref_img_path = "./64.bmp";
const char* output_dir = "./holograms/host/far_holo";
char output_dir_num[100];
unsigned char header_buf[1078];
char output_path[100];

const int height = 128;
const int width = 128;
const int depth = 64;
const int particle_num = 10;
const int image_count = 10;

//All units provided by micro meter
const float mean_diam = 50.0;
const float sd_diam = 10.0;
const float dist_to_cam = 50000.0;

const float peak_bright = 127.0;

void particle_diam_dist (float *diam, int index);
void particle_posi_dist (float x[particle_num], float y[particle_num], float z[particle_num]);
void z_axis_sort (float *array, int left, int right);

void S_fft (float *ak, float *bk, int N, int ff);
void twoDimFFT(float *re, float *im, int flag);
void trans_func(float *re, float *im,float dist);

void particle_volume(float diam[particle_num], float posi_x[particle_num], float posi_y[particle_num], float posi_z[particle_num], unsigned char *volume);
void extract_plane_from_vol (unsigned char *volume, float *plane, int num);
void plane_complex_multiple(float *re1,float *im1, float *re2, float *im2, float *re3, float *im3);
void holo_to_float_image(float *re, float *im, float image[height/2][width/2]);
void plane_initialize (float *plane);

FILE* fp;
/**********************************main***********************************/
int main(int argc, char** argv){
    printf("%s Starting...\n", argv[0]);

    srand((unsigned int)time(NULL));
    sprintf(output_dir_num,"%s/num_%05d/",output_dir,particle_num);
    printf("%s\n",output_dir_num);
    mkdir(output_dir_num, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    fp = fopen(ref_img_path,"rb");
    if(fp == NULL){
        printf("NO REFERENCE IMAGE! quitting...\n");
        exit(1);
    }
    fread(header_buf, sizeof(unsigned char), 1078, fp);
    fclose(fp);

    float diam[particle_num];
    float posi_x[particle_num], posi_y[particle_num], posi_z[particle_num];
    unsigned char *volume;
    float *re_object, *im_object, *re_holo, *im_holo, *re_trans, *im_trans;
    float float_image[(int)height/2][(int)width/2];
    unsigned char image_out[(int)height/2][(int)width/2];
    float dist_to_next_holo;

    re_object = (float *)malloc(sizeof(float)*height*width);
    im_object = (float *)malloc(sizeof(float)*height*width);
    re_trans = (float *)malloc(sizeof(float)*height*width);
    im_trans = (float *)malloc(sizeof(float)*height*width);
    re_holo = (float *)malloc(sizeof(float)*height*width);
    im_holo = (float *)malloc(sizeof(float)*height*width);
    volume = (unsigned char *)malloc(sizeof(unsigned char)*height*width*particle_num);
    
    for (int count = 0; count < image_count; count++){
        particle_posi_dist(posi_x,posi_y,posi_z);
        particle_diam_dist(diam,particle_num);
        z_axis_sort(posi_z,0,particle_num-1);
        particle_volume(diam,posi_x,posi_y,posi_z, volume);
        extract_plane_from_vol(volume, re_object, 0);
        plane_initialize(im_object);

        for (int itr = 0; itr < particle_num-1; itr++){
            twoDimFFT(re_object,im_object,1);
            dist_to_next_holo = posi_z[itr+1] - posi_z[itr];
            trans_func(re_trans,im_trans,dist_to_next_holo);
            plane_complex_multiple(re_object,im_object,re_trans,im_trans,re_holo,im_holo);
            twoDimFFT(re_holo, im_holo, -1);
            extract_plane_from_vol(volume, re_object, itr+1);
            plane_initialize(im_object);
            plane_complex_multiple(re_object,im_object,re_holo,im_holo,re_object,im_object);
        }
        twoDimFFT(re_object,im_object,1);
        trans_func(re_trans,im_trans,dist_to_cam + (float)depth*DX - posi_z[particle_num-1]);
        plane_complex_multiple(re_object,im_object,re_trans,im_trans,re_holo,im_holo);
        twoDimFFT(re_holo,im_holo,-1);

        // //TEST
        // twoDimFFT(re_object,im_object,1);
        // twoDimFFT(re_object,im_object,-1);
        //         twoDimFFT(re_object,im_object,1);
        // twoDimFFT(re_object,im_object,-1);
        // holo_to_float_image(re_object,im_object,float_image);

        holo_to_float_image(re_holo,im_holo,float_image);

        for (int y = 0; y < height/2; y++){
            for (int x = 0; x < width/2; x++){
                image_out[y][x] = (unsigned char)(peak_bright*float_image[y][x]);
            }
        }
    
        sprintf(output_path, "%s/%05d.bmp",output_dir_num,count);
        printf("%s\n",output_path);
        fp = fopen(output_path, "wb");
        fwrite(header_buf, sizeof(unsigned char), 1078, fp);
        fwrite(image_out, sizeof(unsigned char), width*height/4, fp);
        fclose(fp);
        printf("%d of 1000 has processed\n",count);
    }

    free(re_object);
    free(im_object);
    free(re_trans);
    free(im_trans);
    free(re_holo);
    free(im_holo);
    free(volume);
    return 0;
}

void particle_posi_dist (float x[particle_num], float y[particle_num], float z[particle_num]){
    for (int i = 0; i < particle_num; i++) {
        x[i] = (rand() % (int)width/2 + width/4.0) * DX;
        y[i] = (rand() % (int)height/2 + height/4.0) * DX;
        z[i] = (rand() % depth) * DX;
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

void trans_func(float *re, float *im,float dist){
    float tmp, C[3];
    C[0] = 2.0*M_PI*dist/WAVE_LEN;
    C[1] = WAVE_LEN*WAVE_LEN/width/width/DX/DX;
    C[2] = WAVE_LEN*WAVE_LEN/height/height/DX/DX;

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){   
            tmp = C[0]*sqrt(1.0-C[1]*(j-width/2.0)*(j-width/2.0)-C[2]*(i-height/2.0)*(i-height/2.0));
            re[width*i + j] = cos(tmp);
            im[width*i + j] = sin(tmp);
        }
    }
}

void twoDimFFT(float *re, float *im, int flag){
    float re_temp1[width], im_temp1[width], re_temp2[height], im_temp2[height];

    if((flag != 1) && (flag != -1)){
        printf("flag of FFT must be either 1 or -1. Software quitting... \n");
        exit(1);
    }
    
    float *re_array, *im_array;
    re_array = (float *)malloc(sizeof(float)*height*width);
    im_array = (float *)malloc(sizeof(float)*height*width);
    
    if(flag == -1){
        for (int i = 0; i < height/2; i++){
            for (int j = 0; j < width/2; j++){
                re_array[width*i + j] = re[width*(i + height/2) + j + width/2];
                im_array[width*i + j] = im[width*(i + height/2) + j + width/2];
                re[width*(i + height/2) + j + width/2] = re[width*i + j];
                im[width*(i + height/2) + j + width/2] = im[width*i + j];
                re[width*i + j] = re_array[width*i + j];
                im[width*i + j] = im_array[width*i + j];
            }
        }

        for (int i = height/2; i < height; i++){
            for (int j = 0; j < width/2; j++){
                re_array[width*i + j] = re[width*(i - height/2) + j + width/2];
                im_array[width*i + j] = im[width*(i - height/2) + j + width/2];
                re[width*(i - height/2) + j + width/2] = re[width*i + j];
                im[width*(i - height/2) + j + width/2] = im[width*i + j];
                re[width*i + j] = re_array[width*i + j];
                im[width*i + j] = im_array[width*i + j];
            }
        }
    }

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            re_temp1[j] = re[width*i + j];
            im_temp1[j] = im[width*i + j];
        }
        S_fft(re_temp1,im_temp1,width,flag);
        for (int j = 0; j < width; j++){
            re[width*i + j] = re_temp1[j];
            im[width*i + j] = im_temp1[j];
        }
    }

    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            re_temp2[j] = re[width*j + i];
            im_temp2[j] = im[width*j + i];
        }
        S_fft(re_temp2,im_temp2,height,flag);
        for (int j = 0; j < height; j++)
        {
            re[width*j + i] = re_temp2[j];
            im[width*j + i] = im_temp2[j];
        }
    }

    if(flag == 1){
        for (int i = 0; i < height/2; i++){
            for (int j = 0; j < width/2; j++){
                re_array[width*i + j] = re[width*(i + height/2) + j + width/2];
                im_array[width*i + j] = im[width*(i + height/2) + j + width/2];
                re[width*(i + height/2) + j + width/2] = re[width*i + j];
                im[width*(i + height/2) + j + width/2] = im[width*i + j];
                re[width*i + j] = re_array[width*i + j];
                im[width*i + j] = im_array[width*i + j];
            }
        }

        for (int i = height/2; i < height; i++){
            for (int j = 0; j < width/2; j++){
                re_array[width*i + j] = re[width*(i - height/2) + j + width/2];
                im_array[width*i + j] = im[width*(i - height/2) + j + width/2];
                re[width*(i - height/2) + j + width/2] = re[width*i + j];
                im[width*(i - height/2) + j + width/2] = im[width*i + j];
                re[width*i + j] = re_array[width*i + j];
                im[width*i + j] = im_array[width*i + j];
            }
        }
    }
    
    free(re_array);
    free(im_array);
}

void S_fft(float *ak, float *bk, int N, int ff){
    int i,j,k,k1,num,nhalf,phi,phi0,rot[N];
    float s,sc,c,a0,b0,tmp;

    for (i = 0; i < N; i++){
        rot[i] = 0;
    }

    nhalf = N/2;
    num = N/2;
    sc = 2.0*M_PI/(float)N;

    while(num>=1){
        for (j = 0; j < N; j+=(2*num)){
            phi = rot[j]/2;
            phi0 = phi + nhalf;
            c = cos(sc*(float)phi);
            s = sin(sc*(float)(phi*ff));

            for(k=j; k<j+num; k++){
                k1 = k+num;
                a0 = ak[k1]*c - bk[k1]*s;
                b0 = ak[k1]*s + bk[k1]*c;
                ak[k1] = ak[k]-a0;
                bk[k1] = bk[k]-b0;
                ak[k] +=a0;
                bk[k] += b0;
                rot[k] = phi;
                rot[k1] = phi0;
            }
        }
        num /= 2;
    }

    if(ff<0){
        for ( i = 0; i < N; i++){
            ak[i] /= (float)N;
            bk[i] /= (float)N;
        }
    }

    for(i=0; i<N-1; i++){
        if((j=rot[i])>i){
            tmp=ak[i]; ak[i] = ak[j]; ak[j]=tmp;
            tmp=bk[i]; bk[i] = bk[j]; bk[j]=tmp;
        }
    }
}

void particle_volume(float diam[particle_num], float posi_x[particle_num], float posi_y[particle_num], float posi_z[particle_num], unsigned char *volume){
    for (int z = 0; z < particle_num; z++){
        for (int  y = 0; y < height; y++){
            for (int  x = 0; x < width; x++){
                if(((float)x*DX - posi_x[z])*((float)x*DX - posi_x[z]) + ((float)y*DX - posi_y[z])*((float)y*DX - posi_y[z]) <= diam[z]*diam[z]/4.0){
                    volume[width*height*z + width*y + x] = 0;
                }else{
                    volume[width*height*z + width*y + x] = 1;
                }
            }
        }
    }
}

void extract_plane_from_vol (unsigned char *volume, float *plane, int num){
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            plane[width*y + x] = (float)volume[width*height*num + width*y + x];
        }
    }
}

void plane_complex_multiple(float *re1,float *im1, float *re2, float *im2, float *re3, float *im3){
    float tmp1, tmp2;
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            tmp1 = re1[width*y + x]*re2[width*y + x] - im1[width*y + x]*im2[width*y + x];
            tmp2 = re1[width*y + x]*im2[width*y + x] + re2[width*y + x]*im1[width*y + x];
            re3[width*y + x] = tmp1;
            im3[width*y + x] = tmp2;
        }
    }
}

void holo_to_float_image(float *re, float *im, float image[height/2][width/2]){
    for (int y = 0; y < height/2; y++){
        for (int x = 0; x < width/2; x++){
            image[y][x] = sqrt(re[width*(y + height/4) + x + width/4]*re[width*(y + height/4) + x + width/4] + im[width*(y + height/4) + x + width/4]*im[width*(y + height/4) + x + width/4]);
        }
    }
}

void plane_initialize (float *plane){
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            plane[width*y + x] = 0.0;
        }
    }
}