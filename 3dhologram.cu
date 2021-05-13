/*************************************************************************
* Dai NAKAI
* Three dimensional Droplets distribution and its hologram generator
* 2021/5/13
* 
*  Single thread has 512 depth pixels, BlockIdx.x and BlockIdx.y
* corresponds to each cross-section 2-D distribution. 
**************************************************************************/
#include <bits/stdc++.h>
#include <cstdlib>
#include <sys/stat.h>
#include <time.h>
using namespace std;

const int height = 512;
const int width = 512;
const int depth = 512;
const int particle_num = 1000;
const float mean_diam = 80.0;
const float sd_diam = 10;
const float dx = 10.0;
const float wave_len = 0.6328;

const char* output_dir = "./holograms";

float *host_diam, *dev_diam;
float *host_posi_x, *host_posi_y, *host_posi_z;
float *dev_posi_x, *dev_posi_y, *dev_posi_z;
float *host_V, *dev_V;

void particle_diam_dist (float *diam);
void particle_posi_dist (float *x, float *y, float*z);

FILE *fp;
/**********************************main***********************************/
int main(){
    mkdir(output_dir, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    srand((unsigned int)time(NULL));

    host_diam = (float *)malloc(sizeof(float)*particle_num);
    particle_diam_dist(host_diam);

    host_posi_x = (float *)malloc(sizeof(float)*width);
    host_posi_y = (float *)malloc(sizeof(float)*height);
    host_posi_z = (float *)malloc(sizeof(float)*depth);
    particle_posi_dist(host_posi_x, host_posi_y, host_posi_z);



    return 0;
}

void particle_posi_dist (float *x, float *y, float*z){
    for (int i = 0; i < particle_num; i++) {
        x[i] = rand() % width;
        y[i] = rand() % height;
        z[i] = rand() % depth;
    }
}

void particle_diam_dist (float *diam){
    float tmp1, tmp2, tmp3;
    for (int i = 0; i < particle_num ; i++) {
        while(1){
            tmp1 = (float)rand() / (float)RAND_MAX;
            tmp2 = (float)rand() / (float)RAND_MAX;
            tmp3 = sqrt(-2.0*log(tmp1))*cos(2.0*M_PI*tmp2);
            diam[i] = sd_diam*tmp3 + mean_diam;
            if (diam[i] > 0.0) break;
        }
    }
}