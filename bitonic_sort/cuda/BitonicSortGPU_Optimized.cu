/***********************************************************
 *  Optimized bitonic sorting on the GPU.
 *
 * Author: Matt Kelly
 ***********************************************************/

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "BitonicSort.h"

#define TILE_SIZE 256

/*
 * Macro to compare and exchange two elements depending on direction
 */
#define COMPARE_EXCHANGE(a,b) { \
	if( direction ^ (a < b) ) { int tmp = a; a = b; b = tmp; } \
}

/*
 * Each thread running this kernel acts as 4 comparators rather than just one
 */
__global__ void SortStepKernel_Optimized_4Compare( int *arrD, int length, int phase, int step ) {

	int t = blockIdx.x*TILE_SIZE + threadIdx.x; // Global thread index

	step >>= 1;
	// int low = t & (step - 1);
	int i = ((t - (t & (step - 1))) << 2) + (t & (step - 1));
	int direction = (i & phase) == 0;

	if( i + 3*step < length ) {
		int d0 = arrD[i];
		int d1 = arrD[i + step];
		int d2 = arrD[i + 2*step];
		int d3 = arrD[i + 3*step];

		COMPARE_EXCHANGE(d0,d2);
		COMPARE_EXCHANGE(d1,d3);
		COMPARE_EXCHANGE(d0,d1);
		COMPARE_EXCHANGE(d2,d3);

		arrD[i] = d0;
		arrD[i + step] = d1;
		arrD[i + 2*step] = d2;
		arrD[i + 3*step] = d3;
	}

}

/*
 * Each thread running this kernel acts as a single comparator
 */
__global__ void SortStepKernel_Optimized_2Compare( int *arrD, int length, int phase, int step ) {
	
	int t = blockIdx.x*TILE_SIZE + threadIdx.x; // Global thread index

	// int low = t & (step - 1);
	int i = (t << 1) - (t & (step - 1));
	int direction = (i & phase) == 0;

	if( i + step < length ) {
		int d0 = arrD[i +      0];
		int d1 = arrD[i +   step];

		COMPARE_EXCHANGE(d0,d1);

		arrD[i] = d0;
		arrD[i + step] = d1;
	}

}

/* 
 * Host code
 */
bool BitonicSortGPU_Optimized( int *arr, int low, int high, int direction ) {
	
	int *arrD;

	int length = high - low;

	// Copy array to device
	cudaMalloc( (void**) &arrD, length * sizeof(int) );
	cudaMemcpy( arrD, arr, length * sizeof(int), cudaMemcpyHostToDevice );

	dim3 dimBlock(TILE_SIZE, 1);
	dim3 dimGrid((int)ceil((float)length / (float)TILE_SIZE), 1);

	// numPhases = log2(length)
	//int numPhases = (int)floor(log((float)length) / log(2.0));

	/* 
	 * Phases are the "major" stages that count up
	 * Steps are the "minor" stages within a phase that count down
	 * In the optimized version, we can eliminate many kernel launches by
	 * enabling each thread to act as multiple comparators when possible.
	 */
	for( int phase = 2; phase <= length; phase <<= 1 ) {

		int step = phase >> 1;
		while( step > 0 ) {

			int stepShifter = 0;

			if( step >= 2 && stepShifter == 0) {
				SortStepKernel_Optimized_4Compare<<< dimGrid, dimBlock >>> (arrD, length, phase, step);
				stepShifter = 2;
			} else if( stepShifter == 0 ) {
				SortStepKernel_Optimized_2Compare<<< dimGrid, dimBlock >>> (arrD, length, phase, step);
				stepShifter = 1;
			}

			cudaThreadSynchronize();

			step >>= stepShifter;

		}
	}

	// Copy result back to host
	cudaMemcpy( arr, arrD, length * sizeof(int), cudaMemcpyDeviceToHost );

	// Make sure everything worked okay; if not, indicate that error occurred
	cudaError_t error = cudaGetLastError();
	if(error) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		return false;
	} else {
		return true;
	}

}
