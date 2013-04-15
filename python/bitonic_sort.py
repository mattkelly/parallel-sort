#!/usr/bin/env python

import random

def bitonic_sort(up,x):
    if len(x)<=1:
        return x
    else: 
        first = bitonic_sort(True,x[:len(x)/2])
        second = bitonic_sort(False,x[len(x)/2:])
        return bitonic_merge(up,first+second)
 
def bitonic_merge(up,x): 
    # assume input x is bitonic, and sorted list is returned 
    if len(x) == 1:
        return x
    else:
        bitonic_compare(up,x)
        first = bitonic_merge(up,x[:len(x)/2])
        second = bitonic_merge(up,x[len(x)/2:])
        return first + second
 
def bitonic_compare(up,x):
    dist = len(x)/2
    for i in range(dist):  
        if (x[i] > x[i+dist]) == up:
            x[i], x[i+dist] = x[i+dist], x[i] #swap

def main():
    length = 1<<6
    before = list(xrange(0,length)) # list of integers from 1 to 99
    random.shuffle(before)
    print "before: "
    print before
    after = bitonic_sort(True, before)
    print "after: "
    print after

if __name__ == "__main__":
    main()
