/***********************************************************
 *  Bitonic sorting on the CPU.
 *
 * Author: Matt Kelly
 ***********************************************************/

#include <ctime>
#include <cstring>

#include "BitonicSort.h"

#define ITERATIONS 10

/* 
 * Compare two elements and swap depending on direction of sort (CPU)
 */
void BitonicCompareCPU( int *arr, int low, int high, int direction ) {
    if( direction == UP ) {
		// Up
        if( arr[high] < arr[low] ) {
            int tmp = arr[low];
            arr[low] = arr[high];
            arr[high] = tmp; 
        }

    } else {
		// Down
        if( arr[high] > arr[low] ) {
            int tmp = arr[low];
            arr[low] = arr[high];
            arr[high] = tmp; 
        }
    }
}

/* 
 * Merge bitonic array into a monotonic array (CPU)
 */
void BitonicMergeCPU( int *arr, int low, int high, int direction ) {

    if( (high - low) > 1 ) {
        int m = (high - low) / 2;
        for( int i = low; i < low + m; ++i ) {
            BitonicCompareCPU(arr, i, i + m, direction);
        }
		// Merge both halves recursively
        BitonicMergeCPU(arr, low, low + m, direction);
        BitonicMergeCPU(arr, low + m, high, direction);
    }

}

/* 
 * Bitonic Sort on CPU
 */
void BitonicSortCPU( int *arr, int low, int high, int direction ) {

    if( (high - low) > 1 ) {

        int m = (high - low) / 2;
        // Create bitonic array recursively by sorting both halves
        BitonicSortCPU(arr, low, low + m, UP);
        BitonicSortCPU(arr, low + m, high, DOWN);
        // Merge bitonic array to create monotonic array
        BitonicMergeCPU(arr, low, high, direction);

    }

}

/*
 * Count differences between two arrays
 */
int CountDifferences( int *arr1, int *arr2, int length ) {
	int differences = 0;
	for( int i = 0; i < length; ++i ) {
		if( arr1[i] != arr2[i] ) {
			differences++;
			// printf("Element (%d | %d , %d) is different\n", i, arr1[i], arr2[i]);
		}
	}
	return differences;
}

/* 
 * Main
 */
int main( int argc, char **argv ) {

    int *arr; // Array to operate on
	int *cpuResult; // Resulting CPU array
	int *origArray; // Original array to copy back, since multiple iterations are performed
    int length = 1 << 23;

	clock_t startTime, endTime;
	float cpuTime, gpuTimeNaive, gpuTimeOptimized;

	printf("Running Bitonic Sort for %d iterations\n", ITERATIONS);
    printf("Length = %d\n", length);

    arr = (int*) malloc( length * sizeof(int) );
	cpuResult = (int*) malloc( length * sizeof(int) );
	origArray = (int*) malloc( length * sizeof(int) );

    // Fill array with random values
	//printf("Initial\n");
    for( int i = 0; i < length; ++i ) {
		int randVal = rand() % length;
        arr[i] = randVal;
		origArray[i] = randVal;
		arr[i] = i % 2 ? -arr[i] : arr[i];
		//printf("%d ", arr[i]);
    }

    // Perform the CPU sort
	startTime = clock();
	for( int i = 0; i < ITERATIONS; ++i ) {
		memcpy( arr, origArray, length * sizeof(int) );
		BitonicSortCPU( arr, 0, length, UP );
	}
	endTime = clock();

	cpuTime = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;

	// Store CPU result for comparison against GPU
	memcpy( cpuResult, arr, length * sizeof(int) );

    // Print sorted contents
    /*printf("\nSorted (CPU):\n");
    for( int i = 0; i < length; i++ ) {
        printf("%d ", arr[i]);        
    }
    printf("\n");*/

	// Warm-up pass for GPU (Naive)
	memcpy( arr, origArray, length * sizeof(int) );
	BitonicSortGPU_Naive( arr, 0, length, UP );

	// Timed passes for GPU (Naive)
	startTime = clock();
	for( int i = 0; i < ITERATIONS; ++i ) {
		memcpy( arr, origArray, length * sizeof(int) );
		BitonicSortGPU_Naive( arr, 0, length, UP );
	}
	endTime = clock();

	gpuTimeNaive = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;
	int diffNaive = CountDifferences( cpuResult, arr, length );

	// Warm-up pass for GPU (Optimized)
	memcpy( arr, origArray, length * sizeof(int) );
	BitonicSortGPU_Optimized( arr, 0, length, UP );

	// Timed passes for GPU (Optimized)
	startTime = clock();
	for( int i = 0; i < ITERATIONS; ++i ) {
		memcpy( arr, origArray, length * sizeof(int) );
		BitonicSortGPU_Optimized( arr, 0, length, UP );
	}
	endTime = clock();

	gpuTimeOptimized = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;
	int diffOptimized = CountDifferences( cpuResult, arr, length );

	// Print sorted contents
    /*printf("\nSorted (GPU Optimized):\n");
    for( int i = 0; i < length; i++ ) {
        printf("%d ", arr[i]);        
    }
    printf("\n");*/

	printf("\nHost computation took %1.3f ms\n", cpuTime);
	printf("\nDevice computation (naive) took %1.3f ms\n", gpuTimeNaive);
	printf("\nDevice computation (optimized) took %1.3f ms\n", gpuTimeOptimized);

	printf("\nNumber of different elements between CPU and GPU (naive): %d\n", diffNaive);
	printf("\nNumber of different elements between CPU and GPU (optimized): %d\n", diffOptimized);

	// Clean up
	free( origArray );
	free( cpuResult );
	free( arr );

	return 0;

}
