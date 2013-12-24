/***********************************************************
 *  Bitonic sorting on the GPU.
 *
 * Author: Matt Kelly
 ***********************************************************/

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "BitonicSort.h"

#define TILE_SIZE 256

__global__ void SortStepKernel_Naive( int *arrD, int length, int phase, int step ) {

	int tx = threadIdx.x; // Thread index
	int bx = blockIdx.x; // Block index
	int i = bx*TILE_SIZE + tx; // Global thread index

	// int j = i + (subarraySize / 2); // Swap partner index
	int j = i^step; // Swap partner index

	// int subarraySize = 1 << step; // Size of all subarrays
	// int subarrayID = i % subarraySize; // ID of subarray this thread is in

	int direction = ( (i & phase) == 0 );

	// Only lower thread IDs can swap
	if( j > i ) {

		if( direction == UP ) {
			if( arrD[i] > arrD[j] ) {
				int tmp = arrD[i];
				arrD[i] = arrD[j];
				arrD[j] = tmp;
			}
		} else if( direction == DOWN ) {
			if( arrD[i] < arrD[j] ) {
				int tmp = arrD[i];
				arrD[i] = arrD[j];
				arrD[j] = tmp;
			}
		}

	}

}

/* 
 * Host code
 */
bool BitonicSortGPU_Naive( int *arr, int low, int high, int direction ) {
	
	int *arrD;

	int length = high - low;

	// Copy array to device
	cudaMalloc( (void**) &arrD, length * sizeof(int) );
	cudaMemcpy( arrD, arr, length * sizeof(int), cudaMemcpyHostToDevice );

	dim3 dimBlock(TILE_SIZE, 1);
	dim3 dimGrid((int)ceil((float)length / (float)TILE_SIZE), 1);

	// numphases = log2(length)
	int numPhases = (int)floor(log((float)length) / log(2.0));

	/* 
	 * Phases are the "major" stages that count up
	 * Steps are the "minor" stages within a phase that count down
	 */
	for( int phase = 2; phase <= length; phase <<= 1 ) {
		for( int step = phase >> 1; step > 0; step >>= 1 ) {
			SortStepKernel_Naive<<< dimGrid, dimBlock >>> (arrD, length, phase, step);
			cudaThreadSynchronize();
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
