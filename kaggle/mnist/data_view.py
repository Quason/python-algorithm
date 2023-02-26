''' MNIST data viewer
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def csv2img(data):
    img = data.reshape((28, 28))
    return img.astype(np.uint8)


def main():
    csv_fn_train = '/Users/marvin/Documents/kaggle/digit-recognizer/train.csv'
    ds = pd.read_csv(csv_fn_train)
    data = ds.to_numpy()
    img =  csv2img(data[120, 1:])
    print(data[120, 0])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
