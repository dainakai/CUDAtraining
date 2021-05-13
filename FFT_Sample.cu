//Hiroki MATSUSHI
//2018/?/? Make for "GPU lesson"
//2020/07/07 Reconstruct
//2020/07/12 cleaning and Making "CharMaker"
#include <stdio.h>
#include <cufft.h>

const char* input_data_file ="face_padding2.bmp"; //input file name
const char* output_data_file1 ="outputH.bmp"; //output data file
const char* output_data_file2 ="outputFFT-IFFT.bmp"; //output data file
const int data_long=1024; //length of data

unsigned char header_buf[1078];
unsigned char image_in[data_long][data_long];
unsigned char image_out[data_long][data_long];
float ou_frm[data_long][data_long];
float ou_fim[data_long][data_long];
float ou_frmt[data_long][data_long];
float ou_fimt[data_long][data_long];
FILE *input_file; //pointer of input file
FILE *output_file; //pointer of output file

__global__ void fftshift_2D(float *data){
    int data_long_half=data_long/2;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float temp1,temp2;
    unsigned int adr = (x+y*data_long);
    unsigned int adrr =adr+data_long_half*(1+data_long);
    temp1=data[adr];
    temp2=data[adrr];
    data[adr]=temp2;
    data[adrr]=temp1;

    unsigned int adr3 = (x+data_long_half+y*data_long);
    unsigned int adr4=adr3+data_long_half*data_long-data_long_half;
    temp1=data[adr3];
    temp2=data[adr4];
    data[adr3]=temp2;
    data[adr4]=temp1;
}

__global__ void CharMaker(float *input,unsigned char *output){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int adr = (x+y*data_long);
    output[adr]=(unsigned char)input[adr];
}

int main()
{
    /***** Input *****/
        input_file = fopen ( input_data_file ,"rb");if( input_file == NULL ){ printf("I can't open infile!\n");return(0);}
        fread ( header_buf , sizeof ( unsigned char ) ,1078 , input_file );
        fread ( image_in , sizeof ( image_in ),1, input_file);
        fclose ( input_file);

    /***** FFT *****/
        cufftHandle plan;
        cufftComplex *host;
        cufftComplex *dev;

        host=(cufftComplex*)malloc(sizeof(cufftComplex)*data_long*data_long);
        for(int i=0;i<data_long;i++){
            for(int j=0;j<data_long;j++){
                    host[j+i*data_long]=make_cuComplex((float)image_in[i][j],0.0);
            }
        }

        cudaMalloc((void**)&dev,sizeof(cufftComplex)*data_long*data_long);
        cufftPlan2d(&plan,data_long,data_long,CUFFT_C2C);
        cudaMemcpy(dev,host,sizeof(cufftComplex)*data_long*data_long,cudaMemcpyHostToDevice);
        cufftExecC2C(plan,dev,dev,CUFFT_FORWARD);
        cudaMemcpy(host,dev,sizeof(cufftComplex)*data_long*data_long,cudaMemcpyDeviceToHost);

	/***** spect *****/
        //Max and min
        ///This "Max and min" area is a high potential of GPU-programing effect, I think.
        ///Would you like to try programming this area for GPU (Question mark)
        double i_max=0.0,i_min=9999.0,i_sqrt;
        for(int l=0;l<data_long;l++){
            for(int k=0;k<data_long;k++){
                ou_frm[l][k]=cuCrealf(host[k+l*data_long]);
                ou_fim[l][k]=cuCimagf(host[k+l*data_long]);
                i_sqrt=sqrt(ou_frm[l][k]*ou_frm[l][k]+ou_fim[l][k]*ou_fim[l][k]);
                ou_frmt[k][l]=log10(i_sqrt);
                if(i_sqrt==0.0){ou_frmt[k][l]=0.0;};
                if(ou_frmt[k][l]<i_min){i_min=ou_frmt[k][l];};
                if(ou_frmt[k][l]>i_max){i_max=ou_frmt[k][l];};
            }
        }

        for(int l=0;l<data_long;l++){
            for(int k=0;k<data_long;k++){
                if(ou_frm[k][l]!=0.0){ou_frm[k][l]=(ou_frmt[k][l]-i_min)/(i_max-i_min)*255.0;}
                if(ou_frm[k][l]==0){ou_frm[k][l]=255.0;};
            }
        }

        //rearrangement
        const int data_long_half=data_long/2;
        int blocksize = 16; 
        dim3 block2 (blocksize, blocksize, 1);
        dim3 grid2  (data_long_half/block2.x,data_long_half/block2.y, 1);
        float *dev_F;
        cudaMalloc((void**)&dev_F,sizeof(float)*data_long*data_long);//float is correct

        cudaMemcpy(dev_F,ou_frm,sizeof(float)*data_long*data_long,cudaMemcpyHostToDevice);//float is correct
        fftshift_2D<<<grid2,block2>>>(dev_F);

        //"float" is changed "unsigned char"
        dim3 block1 (blocksize, blocksize, 1);
        dim3 grid1  (data_long/block2.x,data_long/block2.y, 1);
        unsigned char *dev_UnChar;
        cudaMalloc((void**)&dev_UnChar,sizeof(unsigned char)*data_long*data_long);

        CharMaker<<<grid1,block1>>>(dev_F,dev_UnChar);
        cudaMemcpy(image_out,dev_UnChar,sizeof(unsigned char)*data_long*data_long,cudaMemcpyDeviceToHost);

        //output
        output_file=fopen(output_data_file1,"wb");
        fwrite ( header_buf , sizeof ( unsigned char ) ,1078 , output_file );
        fwrite ( image_out , sizeof ( image_out ),1, output_file );
        fclose(output_file);

    /**** IFFT ****/
        cufftExecC2C(plan,dev,dev,CUFFT_INVERSE);
        cudaMemcpy(host,dev,sizeof(cufftComplex)*data_long*data_long,cudaMemcpyDeviceToHost);

        ///This area is a high potential of GPU-programing effect, I think.
        ///Would you like to try programming this area for GPU (Question mark)
        for(int l=0;l<data_long;l++){
            for(int k=0;k<data_long;k++){
            image_out[k][l]=(unsigned char)(cuCrealf(host[l+k*data_long])/((double)data_long*(double)data_long));
            }
        }

        //output
        output_file=fopen(output_data_file2,"wb");
        fwrite ( header_buf , sizeof ( unsigned char ) ,1078 , output_file );
        fwrite ( image_out , sizeof ( image_out ),1, output_file );
        fclose(output_file);

    /**** finale ****/
	cufftDestroy(plan);
	cudaFree(dev);
	free(host);
}
