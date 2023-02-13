'''
    算法: 冒泡排序
    时间复杂度: O(N^2)
'''
import numpy as np


def bubble_sort(x):
    for i in range(len(x) - 1):
        for j in range(i+1, len(x)):
            if x[i] > x[j]:
                n = x[i]
                x[i] = x[j]
                x[j] = n
    return x


if __name__ == '__main__':
    x = np.random.rand(10)
    print(x)
    print(bubble_sort(x))
