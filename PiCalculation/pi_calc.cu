#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define TRIALS_PER_THREAD 1024
#define BLOCKS 256
#define THREADS 256


__global__ void gpuPiCalculate(float *estimate, curandState *states) {
	unsigned long id = threadIdx.x + blockDim.x * blockIdx.x;
	int pointsInCircle = 0;
	float x, y;

	curand_init(id, id, 0, &states[id]);  //initialize curand

	for (int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform(&states[id]);
		y = curand_uniform(&states[id]);
		pointsInCircle += (x*x + y * y <= 1.0f); 
	}
	estimate[id] = 4.0f * pointsInCircle / (float)TRIALS_PER_THREAD;
}

float cpuPiCalculate(long trials) {
	float x, y;
	long pointsInCircle = 0;
	for (long i = 0; i < trials; i++) {
		x = rand() / (float)RAND_MAX;
		y = rand() / (float)RAND_MAX;
		pointsInCircle += (x * x + y * y <= 1.0f);
	}
	return 4.0f * pointsInCircle / trials;
}

int main(int argc, char *argv[]) {
	clock_t start, stop;
	float host[BLOCKS * THREADS];
	float *dev;
	curandState *devStates;

	start = clock();
	cudaMalloc((void **)&dev, BLOCKS * THREADS * sizeof(float)); 
	cudaMalloc((void **)&devStates, THREADS * BLOCKS * sizeof(curandState));

	gpuPiCalculate << <BLOCKS, THREADS >> > (dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);
	float gpuPI = 0;
	for (int i = 0; i < BLOCKS * THREADS; i++) {
		gpuPI += host[i];
	}
	gpuPI /= (BLOCKS * THREADS);
	stop = clock();
	printf("GPU PI= %f\n", gpuPI);
	printf("GPU PI calculate time %f s.\n", (stop - start) / (float)CLOCKS_PER_SEC);
	
	start = clock();
	float cpuPI = cpuPiCalculate(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	printf("CPU PI= %f\n", cpuPI);
	printf("CPU PI calculate time %f s.\n", (stop - start) / (float)CLOCKS_PER_SEC);

	return 0;
}
