#include <math.h>
#include <fstream>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
using namespace std;
#define DEBUG
#define PINNED

bool readFile(int16_t* read, const char* filePath);
bool saveFile(int16_t* write, const char* filePath);


float max_distance = 0;
int HEIGHT = 240;
int WIDTH = 288;
timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}


 __global__ 
void calculate_atan2_cu(int16_t* phase1_cu, int16_t* phase2_cu,
            int16_t* phase3_cu, int16_t* phase4_cu,
            int16_t* depth_out_cu, int16_t* confidence_out_cu,int N){

    const int index = blockIdx.x*blockDim.x + threadIdx.x; 

    if(index < N){ 

                int16_t  phase_1  = *(phase1_cu + index);
                int16_t  phase_2  = *(phase2_cu + index);
                int16_t  phase_3  = *(phase3_cu + index);
                int16_t  phase_4  = *(phase4_cu + index);


                //printf("%d,%d,%d,%d\n",phase_1,phase_2,phase_3,phase_4);

                auto phase1_t = phase_1 >> 4;
                auto phase2_t = phase_2 >> 4;
                auto phase3_t = phase_3 >> 4;
                auto phase4_t = phase_4 >> 4;

                int16_t I = phase1_t - phase3_t;
                int16_t Q = phase2_t - phase4_t;

                int out_index = index; 

                   
                confidence_out_cu[out_index] = sqrt(float(I*I + Q*Q));
                //printf("rsqrt(%d)  confidence_out_cu[%d]=%d",I*I+Q*Q,index,confidence_out_cu[index]);

                float angle = atan(-(float)Q/(float)I);
                depth_out_cu[out_index] = (angle + M_PI) * 3.7474 * 1000 / (2 * M_PI);
    }
}


int main(int argc, char *argv[])
{
	const char *path=NULL;

	printf("\n");

	printf("Usage: %s relative-path-for-phase-data\n",argv[0]);
	printf("depth_cu and confidence2_cu are put in the same path with %s\n",argv[0]);
	printf("****Be noticed that the calculation will be done for 1000 times****\n");
	printf("If you want to profile %s, try ",argv[0]);
	printf("sudo /usr/local/cuda/bin/nvprof %s\n",argv[0]);
	printf("otherwise, just run %s\n",argv[0]);

	printf("\n");

	if(argc == 1){
		printf("Now loading phase data from ./data/originalData/\n");
		path = "./data/originalData/";
	}
	else{
		printf("now loading phase data from %s\n",argv[1]);
		path = argv[1];
	}
    const int nStreams = 4;
    timespec time1, time2;
   cudaStream_t stream[nStreams];

    for (int i = 0; i < nStreams; ++i)
        checkCuda( cudaStreamCreate(&stream[i]) );



    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&dummyEvent) );



    int fmod = 40;
    max_distance = 0.5*299792458/(double)fmod/1000;

    int16_t* phase1;
    int16_t* phase2;
    int16_t* phase3;
    int16_t* phase4;
    int16_t* depth1_out; 
    int16_t* confidence1_out;

#ifndef PINNED 
    phase1 = new int16_t[WIDTH*HEIGHT];
    phase2 = new int16_t[WIDTH*HEIGHT];
    phase3 = new int16_t[WIDTH*HEIGHT];
    phase4 = new int16_t[WIDTH*HEIGHT];
    depth1_out = new int16_t[WIDTH*HEIGHT]; 
    confidence1_out = new int16_t[WIDTH*HEIGHT];
#else
    checkCuda( cudaMallocHost((void**)&phase1, sizeof(int16_t)*WIDTH*HEIGHT) );      // host pinned
    checkCuda( cudaMallocHost((void**)&phase2, sizeof(int16_t)*WIDTH*HEIGHT) );      // host pinned
    checkCuda( cudaMallocHost((void**)&phase3, sizeof(int16_t)*WIDTH*HEIGHT) );      // host pinned
    checkCuda( cudaMallocHost((void**)&phase4, sizeof(int16_t)*WIDTH*HEIGHT) );      // host pinned
    checkCuda( cudaMallocHost((void**)&depth1_out,      sizeof(int16_t)* WIDTH*HEIGHT) );      // host pinned
    checkCuda( cudaMallocHost((void**)&confidence1_out, sizeof(int16_t)* WIDTH*HEIGHT) );      // host pinned
#endif


    int16_t *phase1_cu[nStreams],*phase2_cu[nStreams],*phase3_cu[nStreams],*phase4_cu[nStreams],*depth1_out_cu[nStreams],*confidence1_out_cu[nStreams]; 

  for(int i=0;i<nStreams;i++){
        checkCuda(cudaMalloc((void**)&phase1_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));
        checkCuda(cudaMalloc((void**)&phase2_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));
        checkCuda(cudaMalloc((void**)&phase3_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));
        checkCuda(cudaMalloc((void**)&phase4_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));

        checkCuda(cudaMalloc((void**)&depth1_out_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));
        checkCuda(cudaMalloc((void**)&confidence1_out_cu[i], sizeof(int16_t) * WIDTH * HEIGHT));
    }




   char *file = new char[ strlen(path)+ 10];

   memset(file,0,strlen(path)+ 10); 

   strcat(file,path);


   if(!readFile(phase1, strcat(file,"phase0")) || (*(file+strlen(path)) = '\0')||
           !readFile(phase2, strcat(file,"phase90")) ||(*(file+strlen(path)) = '\0')||
           !readFile(phase3, strcat(file,"phase180")) ||(*(file+strlen(path)) = '\0')||
           !readFile(phase4, strcat(file,"phase270")) )
   {
       printf("read failed!\n");
       return 0;
   }

    printf("calculation is started\n");

    float ms; // elapsed time in milliseconds

    checkCuda( cudaEventRecord(startEvent,stream[nStreams-1]) );
    clock_gettime(CLOCK_MONOTONIC, &time1);
    for(int i=0;i<1000;i+=nStreams){
        // Transfer data from host to device memory
        for(int j=0;j<nStreams;j++){
            checkCuda(cudaMemcpyAsync(phase1_cu[j],phase1, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice,stream[j]));
            checkCuda(cudaMemcpyAsync(phase2_cu[j],phase2, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice,stream[j]));
            checkCuda(cudaMemcpyAsync(phase3_cu[j],phase3, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice,stream[j]));
            checkCuda(cudaMemcpyAsync(phase4_cu[j],phase4, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice,stream[j]));
        }

        for(int j=0;j<nStreams;j++){
            calculate_atan2_cu<<<540,128,0,stream[j]>>>(phase1_cu[j],phase2_cu[j],phase3_cu[j],phase4_cu[j],depth1_out_cu[j],confidence1_out_cu[j], WIDTH*HEIGHT);
        }
        for(int j=0;j<nStreams;j++){
            checkCuda(cudaMemcpyAsync(depth1_out, depth1_out_cu[j],sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost,stream[j]));
            checkCuda(cudaMemcpyAsync(confidence1_out,confidence1_out_cu[j], sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost,stream[j]));
        }
    }
    checkCuda( cudaEventRecord(stopEvent, stream[nStreams-1]) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );


    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &time2);
    printf("Time for multiple stream transfer and execute (ms): %f    ", ms);
    cout<<diff(time1,time2).tv_sec<<":"<<diff(time1,time2).tv_nsec<<endl;

    printf("stream with %d calculation is done\n",nStreams);



    clock_gettime(CLOCK_MONOTONIC, &time1);
    checkCuda( cudaEventRecord(startEvent,0) );
    for(int i=0;i<1000;i++){
        // Transfer data from host to device memory
        cudaMemcpy(phase1_cu[0],phase1, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
        cudaMemcpy(phase2_cu[0],phase2, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
        cudaMemcpy(phase3_cu[0],phase3, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
        cudaMemcpy(phase4_cu[0],phase4, sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

        calculate_atan2_cu<<<540,128>>>(phase1_cu[0],phase2_cu[0],phase3_cu[0],phase4_cu[0],depth1_out_cu[0],confidence1_out_cu[0], WIDTH*HEIGHT);

        cudaMemcpy(depth1_out, depth1_out_cu[0],sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        cudaMemcpy(confidence1_out,confidence1_out_cu[0], sizeof(int16_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    }

    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &time2);
    printf("Time for NULL stream transfer and execute (ms): %f     ", ms);
    cout<<diff(time1,time2).tv_sec<<":"<<diff(time1,time2).tv_nsec<<endl;




   printf("Calculation is done and saving depth and confidence!\n");
   saveFile(depth1_out, "depth_cu");
   saveFile(confidence1_out, "confidence2_cu");
   printf("\n");
   return 0;
}


bool readFile(int16_t* read, const char* filePath)
{
//	printf("filePath=%s\n",filePath);
    std::ifstream infile;
    infile.open(filePath);
    if(!infile)
    {
        printf("can not open file :%s\n",filePath);
        return false;
    }

    while(!infile.eof())
    {
        infile>>*read;
        read++;
    }
    infile.close();
    return true;
}

bool saveFile(int16_t* write, const char* filePath)
{
    std::ofstream fout(filePath);

    for(auto j=0; j < HEIGHT; ++j)
    {
        for(auto i =0; i < WIDTH; i++)
        {
            fout << write[j*WIDTH+i] << "\t" ;
        }
        fout << "\n";
    }
    fout.close();
}
