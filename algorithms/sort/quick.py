'''
    算法: 快速排序
    时间复杂度: O(N*logN)
    涉及到递归, 并没有太看懂...(基本思想是分治法, 每次选择一个数作为pivot, 比他小的放左边, 比他大的放右边)
'''
import numpy as np


def _partition(x, low, high):
    pivot = x[high]
    i = low - 1
    for j in range(low, high):
        if x[j] <= pivot:
            i = i + 1
            (x[i], x[j]) = (x[j], x[i])
    (x[i + 1], x[high]) = (x[high], x[i + 1])
    return i + 1


def quick_sort(x, low, high):
    if low < high:
        pi = _partition(x, low, high)
        quick_sort(x, low, pi - 1)
        quick_sort(x, pi + 1, high)


if __name__ == '__main__':
    x = np.random.rand(10)
    print(x)
    quick_sort(x, 0, len(x)-1)
    print(x)
