''' scikit-learn algorithms
'''
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


def dataset(csv_fn, is_train):
    ds = pd.read_csv(csv_fn)
    data = ds.to_numpy()
    if is_train:
        labels = data[:, 0]
        pixels = data[:, 1:]
        return pixels, labels
    else:
        pixels = data
        return pixels


def train(train_fn):
    pixels, labels = dataset(train_fn, is_train=True)
    x_train, x_test, y_train, y_test = train_test_split(
        pixels, labels, test_size=0.3, shuffle=True)
    
    # # SVM (0.98)
    # clf = svm.SVC(
    #     C=10,
    #     kernel="rbf",  # rbf is better
    #     degree=3,
    #     gamma="scale",
    #     coef0=0.0,
    #     shrinking=True,
    #     probability=False,
    #     tol=1e-3,
    #     cache_size=200,
    #     class_weight=None,
    #     verbose=False,
    #     max_iter=-1,
    #     decision_function_shape="ovo",  # ovr or ovo
    #     break_ties=False,
    #     random_state=None,
    # )

    # # RF (0.93)
    # clf = RandomForestClassifier(
    #     n_estimators=10,
    #     criterion="gini",
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     min_weight_fraction_leaf=0.0,
    #     max_features="sqrt",
    #     max_leaf_nodes=None,
    #     min_impurity_decrease=0.0,
    #     bootstrap=True,
    #     oob_score=False,
    #     n_jobs=None,
    #     random_state=None,
    #     verbose=0,
    #     warm_start=True,
    #     class_weight=None,
    #     ccp_alpha=0.0,
    #     max_samples=None
    # )

    # MLP (0.96)
    clf = MLPClassifier(
        hidden_layer_sizes=(100,100),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    )

    clf.fit(x_train, y_train)
    y_test_predict = clf.predict(x_test)
    y_train_predict = clf.predict(x_train)
    oa_train = len(y_train[y_train==y_train_predict]) / len(y_train)
    oa_test = len(y_test[y_test==y_test_predict]) / len(y_test)
    print(f'train OA: {oa_train}')
    print(f'test OA: {oa_test}')
    return clf


def predict(test_fn, clf, dst_fn):
    ds = pd.read_csv(test_fn)
    data = ds.to_numpy()
    predict = clf.predict(data)
    with open(dst_fn, 'w') as fp:
        fp.write('ImageId,Label\n')
        for i in range(len(predict)):
            fp.write(f'{i+1},{int(predict[i])}\n')


if __name__ == '__main__':
    t0 = time.time()
    train_fn = '/Users/marvin/Documents/kaggle/digit-recognizer/train.csv'
    test_fn = '/Users/marvin/Documents/kaggle/digit-recognizer/test.csv'
    dst_fn = '/Users/marvin/Documents/kaggle/digit-recognizer/predict_svm.csv'
    clf = train(train_fn)
    test_predict = predict(test_fn, clf, dst_fn)
    print('OT: %.1fs' % (time.time() - t0))
