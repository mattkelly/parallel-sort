/***********************************************************
 * Definitions for bitonic sorting on the CPU and GPU.
 *
 * Author: Matt Kelly
 ***********************************************************/

#include <cstdlib>
#include <cstdio>

// Direction definitions
#define UP   1
#define DOWN 0

/*
 * CPU function for swapping two elements depending on direction
 */
void BitonicCompareCPU( int *arr, int low, int high, int direction );

/*
 * CPU function (recursive) for merging a bitonic array into a monotonic array
 */
void BitonicMergeCPU( int *arr, int low, int high, int direction );

/*
 * CPU function (recursive) that performs a bitonic sort
 */
void BitonicSortCPU( int *arr, int low, int high, int direction );

/*
 * Count number of differing elements between two arrays
 */
int CountDifferences( int *arr1, int *arr2, int length );

/*
 * GPU function to perform bitonic sort naively (host code portion)
 */
bool BitonicSortGPU_Naive( int *arr, int low, int high, int direction );

/*
 * GPU function to perform bitonic sort optimally (host code portion)
 */
bool BitonicSortGPU_Optimized( int *arr, int low, int high, int direction );
