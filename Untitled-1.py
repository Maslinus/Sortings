import time
import random
from datetime import datetime, timedelta
from numba import njit, prange

import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math

from sys import setrecursionlimit
matplotlib.style.use('ggplot')
setrecursionlimit(1000000)

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# массив цифр

arrDigits100 = [random.randint(0, 9) for _ in range(100)]
arrDigits9000 = [random.randint(0, 9) for _ in range(9000)]
arrDigits30000 = [random.randint(0, 9) for _ in range(30000)]
arrDigits300000 = [random.randint(0, 9) for _ in range(300000)]

# массив чисел

arrNums100 = [random.randint(-1000, 1000) for _ in range(100)]
arrNums9000 = [random.randint(-1000, 1000) for _ in range(9000)]
arrNums30000 = [random.randint(-1000, 1000) for _ in range(30000)]
arrNums300000 = [random.randint(-1000, 1000) for _ in range(300000)]

# массив строк
strLen = 5

arrString100 = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(strLen)) for _ in range(100)]
arrString9000 = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(strLen)) for _ in range(9000)]
arrString30000 = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(strLen)) for _ in range(30000)]
arrString300000 = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(strLen)) for _ in range(300000)]

# массив дат

arrDataTime100 = [datetime(2000, 1, 1) + timedelta(days=random.randint(0, (datetime(2023, 1, 1) - datetime(2000, 1, 1)).days)) for _ in range(100)]
arrDataTime9000 = [datetime(2000, 1, 1) + timedelta(days=random.randint(0, (datetime(2023, 1, 1) - datetime(2000, 1, 1)).days)) for _ in range(9000)]
arrDataTime30000 = [datetime(2000, 1, 1) + timedelta(days=random.randint(0, (datetime(2023, 1, 1) - datetime(2000, 1, 1)).days)) for _ in range(30000)]
arrDataTime300000 = [datetime(2000, 1, 1) + timedelta(days=random.randint(0, (datetime(2023, 1, 1) - datetime(2000, 1, 1)).days)) for _ in range(300000)]

#В худшем случае: n^2, в лучшем случае: n, в среднем: n^2
@njit(fastmath=True)
def StupidSortCPU(a): 
    n=len(a)
    index =0
    iter=0
    while index<n:
        iter+=1
        if index==0:
            index+=1
        if a[index]>=a[index-1]:
            index+=1
        else:
            a[index],a[index-1]=a[index-1],a[index]
            index-=1
    return iter

# пузырек
#В худшем случае: n^2, в лучшем случае: n, в среднем: n^2
@njit(fastmath=True)
def BubbleSortCPU(a): 
    iter=0
    swapped = False
    for i in prange (1,len(a)):
        for j in prange(0,len(a)-i):
            iter+=1
            if a[j]>a[j+1]:
                swapped=True
                a[j],a[j+1]=a[j+1],a[j]
        if not swapped:
            return iter
    return iter

# вставки
#В худшем случае: n^2, в лучшем случае: n, в среднем: n^2
@njit(fastmath=True)
def InsertionSortCPU(a):
    iter=0
    for i in prange(1,len(a)):
        key = a[i]
        j= i-1
        while j>=0 and key<a[j]:
            iter+=1
            a[j+1]=a[j]
            j-=1
        a[j+1]=key
    return iter

# выбором
#В худшем случае: n^2, в лучшем случае: n, в среднем: n^2
@njit(fastmath=True)
def SelectionSortCPU(a):
    iter=0
    for i in prange (len(a)):
        minInd=i
        for j in prange(i+1,len(a)):
            iter+=1
            if a[j]<a[minInd]:
                minInd=j
        a[i],a[minInd]=a[minInd],a[i]
    return iter

# двунаправленная пузырьковая
#В худшем случае: n^2, в лучшем случае: n, в среднем: n^2
@njit(fastmath=True)
def CocktailSortCPU(a):
    iter=0
    swapped = True
    l=0
    r=len(a)-1
    while(swapped==True):
        for i in prange(l, r):
            iter+=1
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        r=r-1
        for i in prange(r-1, l-1, -1):
            iter+=1
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        l=l+1
    return iter

@njit(fastmath=True)
def PartitionCPU(a,low,high):
    pivot=a[high]
    i=low-1
    for j in prange(low,high):
        if a[j]<=pivot:
            i=i+1
            a[i],a[j]=a[j],a[i]
    a[i+1],a[high]=a[high],a[i+1]
    return i+1

# быстрая
#В худшем случае: n^2 В лучшем случае: n*log(n) В среднем: n*log(n)
def QuickSortCPU(a,low,high):
    iter=0
    if low<high:
        iter+=1
        pi=PartitionCPU(a,low,high)
        iter+=QuickSortCPU(a,low,pi-1)
        iter+=QuickSortCPU(a,pi+1,high)
    return iter


@njit(fastmath=True)
def MergeCPU(a,l,m,r):
    iter=0
    n1=m-l+1
    n2=r-m

    L=[0]*(n1)
    R=[0]*(n2)

    for i in prange(0,n1):
        L[i]=a[l+i]
    for j in prange(0,n2):
        R[j] = a[m+1+j]

    i=0
    j=0
    k=l

    while i<n1 and j<n2:
        if L[i]<=R[j]:
            iter+=1
            a[k]=L[i]
            i+=1
        else:
            iter+=1
            a[k]=R[j]
            j+=1
        k+=1

    while i<n1:
        iter+=1
        a[k]=L[i]
        i+=1
        k+=1

    while j<n2:
        iter+=1
        a[k]=R[j]
        j+=1
        k+=1
    return iter

# слияние
#Наихудший случай: O(n*log(n)) Наилучший случай: Omega(n*log(n)) Среднее значение: Theta(n*log(n))
@njit(fastmath=True)
def MergeSortCPU(a,l,r):
    iter=0
    if l<r:
        iter+=1
        m=l+(r-l)//2

        iter+=MergeSortCPU(a,l,m)
        iter+=MergeSortCPU(a,m+1,r)
        iter+=MergeCPU(a,l,m,r)
    return iter

@njit(fastmath=True)
def HeapifyCPU(a,n,i):
    iter=0
    largest= i
    l=2*i+1
    r=2*i+2

    if l<n and a[i]<a[l]:
        iter+=1
        largest=l
    if r<n and a[largest]<a[r]:
        iter+=1
        largest=r
    if largest!=i:
        iter+=1
        a[i],a[largest]=a[largest],a[i]
        iter+=HeapifyCPU(a,n,largest)
    return iter

# куча
#Худший случай: n*log(n) Лучший случай: n*log(n) или n (равные ключи) Среднее значение: n*log(n)
@njit(fastmath=True)
def HeapSortCPU(a):
    iter=0
    n=len(a)
    for i in prange(n//2-1,-1,-1):
        iter+=HeapifyCPU(a,n,i)
    for i in prange(n-1,0,-1):
        a[i],a[0]=a[0],a[i]
        iter+=HeapifyCPU(a,i,0)
    return iter

@njit(fastmath=True)
def CountingSortCPU(array, place):
    iter=0
    size = len(array)
    output = [0] * size
    count = [0] * 10

    for i in prange(0, size):
        iter+=1
        index = array[i] // place
        count[index % 10] += 1

    for i in prange(1, 10):
        iter+=1
        count[i] += count[i - 1]

    i = size - 1
    while i >= 0:
        iter+=1
        index = array[i] // place
        output[count[index % 10] - 1] = array[i]
        count[index % 10] -= 1
        i -= 1

    for i in prange(0, size):
        array[i] = output[i]
    return iter

# порозрядная
# Сложность O(n)
def RadixSortCPU(a):
    iter=0
    if type(a[0])!=type(10):
        print("Can't be used with this data type")
        return
    max_element = max(a)

    place = 1
    while max_element // place > 0:
        iter+=CountingSortCPU(a, place)
        place *= 10
    return iter

@njit(fastmath=True)
def CalculateMinRun(n):
    r=0
    while n>=32:
        r|= n&1
        n>>=1
    return n+r

@njit(fastmath=True)
def LRInsertionSortCPU(a,l,r):
    iter=0
    for i in prange(l+1,r+1):
        j=i
        while j>l and a[j]<a[j-1]:
            iter+=1
            a[j],a[j-1]=a[j-1],a[j]
            j-=1
    return iter

#В худшем случае: n*log(n) В лучшем случае: n В среднем: n*log(n)
@njit(fastmath=True)
def TimSortCPU(a): 
    iter=0
    n=len(a)
    minRun = CalculateMinRun(n)

    for start in prange(0,n,minRun):
        end = min(start+minRun-1,n-1)
        iter+=LRInsertionSortCPU(a,start,end)

    size = minRun
    while size<n:
        for left in prange (0,n,2*size):
            mid = min(n-1,left+size-1)
            right=min((left+2*size-1),(n-1))

            if mid<right:
                iter+=MergeCPU(a,left,mid,right)
        size*=2
    return iter

def sort(Data1):
    Data = Data1
    tIme = [0.0] * 10
    i=0
    start = time.time()
    StupidSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    BubbleSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    InsertionSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1
    
    start = time.time()
    SelectionSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    CocktailSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    
    Data = Data1

    start = time.time()
    QuickSortCPU(Data,0,len(Data)-1)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    
    Data = Data1
 
    start = time.time()
    MergeSortCPU(Data,0,len(Data)-1)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    HeapSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    RadixSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    TimSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    
    print(tIme)
    
def sort2(Data1):
    Data = Data1
    tIme = [0.0] * 6
    i=0
    start = time.time()
    StupidSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    BubbleSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    InsertionSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1
    
    start = time.time()
    SelectionSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1

    Data = Data1

    start = time.time()
    CocktailSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    
    Data = Data1

    start = time.time()
    QuickSortCPU(Data,0,len(Data)-1)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    '''
    Data = Data1
 
    start = time.time()
    MergeSortCPU(Data,0,len(Data)-1)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    '''
    Data = Data1

    start = time.time()
    HeapSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    i+=1
    '''
    Data = Data1

    start = time.time()
    TimSortCPU(Data)
    end = time.time()
    tIme[i]=end - start
    print(tIme[i])
    
    '''
    print(tIme)
    


sort(arrDigits100)
sort(arrDigits9000)
sort(arrDigits30000)
sort(arrDigits300000)

sort(arrNums100)
sort(arrNums9000)
sort(arrNums30000)
sort(arrNums300000)

sort2(arrString100)
sort2(arrString9000)
sort2(arrString30000)
sort2(arrString300000)

sort(arrDataTime100)
sort(arrDataTime9000)
sort(arrDataTime30000)
sort(arrDataTime300000)
