#include <stdlib.h>
#include <stdio.h>

#define UP   1
#define DOWN 0

/* 
 * Bitonic Compare 
 */
void bitonic_compare( int *arr, int low, int high, int direction ) {

    if( direction == UP ) {
        /* up */
        if( arr[high] < arr[low] ) {
            int tmp = arr[low];
            arr[low] = arr[high];
            arr[high] = tmp; 
        }

    } else {
        /* down */
        if( arr[high] > arr[low] ) {
            int tmp = arr[low];
            arr[low] = arr[high];
            arr[high] = tmp; 
        }
    }

}

/* 
 * Bitonic Merge 
 * Forms sorted array from bitonic array
 */
void bitonic_merge( int *arr, int low, int high, int direction ) {

    int i; /* loop index */
    int m; /* halfway index */

    if( (high - low) > 1 ) {
        m = (high - low) / 2;
        for( i = low; i < low + m; ++i ) {
            bitonic_compare(arr, i, i + m, direction);
        }
        bitonic_merge(arr, low, low + m, direction);
        bitonic_merge(arr, low + m, high, direction);
    }

}

/* 
 * Bitonic Sort
 */
void bitonic_sort( int *arr, int low, int high, int direction ) {

    int m; /* halfway index */
    int i;

    if( (high - low) > 1 ) {

        m = (high - low) / 2;
        /* create bitonic array */
        bitonic_sort(arr, low, low + m, UP);
        bitonic_sort(arr, low + m, high, DOWN);
        /* merge bitonic array to create monotonic array */
        bitonic_merge(arr, low, high, direction);

    }

}

/* 
 * Main
 */
int main( int argc, char **argv ) {

    int *arr; /* Array to sort */
    int length = 1 << 10;
    int i; /* loop index */

    printf("Length = %d\n", length);

    arr = (int*) malloc( length * sizeof(int) );

    /* Fill array and print initial contents */
    printf("\nInitial:\n");
    for( i = 0; i < length; ++i ) {
        arr[i] = rand() % length;
        //arr[i] = length-i;
        printf("%d ", arr[i]);        
    }
    printf("\n");

    /* Perform the sort */
    bitonic_sort( arr, 0, length, UP );

    /* Print sorted contents */
    printf("\nSorted:\n");
    for( i = 0; i < length; i++ ) {
        printf("%d ", arr[i]);        
    }
    printf("\n");

}
