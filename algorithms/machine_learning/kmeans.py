''' 最基础的无监督聚类: k-means
'''
import numpy as np
import random
import matplotlib.pyplot as plt


def data_generator():
    class_ce = np.array([[1,1], [5,5], [1,5]])
    points = []
    for i in range(class_ce.shape[0]):
        class_x = list(class_ce[i][0] + np.random.rand(5) - 0.5)
        class_y = list(class_ce[i][1] + np.random.rand(5) - 0.5)
        for j in range(5):
            points.append([class_x[j], class_y[j]])
    random.shuffle(points)
    return np.array(points)


def dist_calculator(x ,y):
    return (x[0] - y[0])**2 + (x[1] - y[1])**2


def classify(data, old_ces):
    data_classes = np.zeros(data.shape[0])
    for i, item in enumerate(data):
        min_dist = np.Inf
        for j, ce in enumerate(old_ces):
            dist = dist_calculator(item, ce)
            if dist < min_dist:
                min_dist = dist
                data_classes[i] = j
    new_ces = []
    sum_dist = 0
    for i in range(old_ces.shape[0]):
        tmp_xy = data[data_classes==i, :]
        new_ce = np.mean(tmp_xy, axis=0)
        new_ces.append(new_ce)
        sum_dist += dist_calculator(old_ces[i], new_ce)
    if sum_dist < 1e-4:
        return np.array(new_ces), data_classes
    else:
        return classify(data, np.array(new_ces))


def kmeans(data, classes=3):
    xmax = np.max(data[:, 0])
    xmin = np.min(data[:, 0])
    ymax = np.max(data[:, 1])
    ymin = np.min(data[:, 1])
    cex_init = np.random.rand(classes) * (xmax - xmin) + xmin
    cey_init = np.random.rand(classes) * (ymax - ymin) + ymin
    ces = np.vstack((cex_init, cey_init)).T
    return classify(data, ces)


if __name__ == '__main__':
    data = data_generator()
    ces, data_classes = kmeans(data, classes=3)
    plt.plot(data[:, 0], data[:, 1], 'b.')
    plt.plot(ces[:, 0], ces[:, 1], 'ro')
    plt.show()
