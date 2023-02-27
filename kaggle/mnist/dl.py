''' deep learning
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset, DataLoader, random_split


class AlexNet(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistDataset(Dataset):
    def __init__(self, csv_fn, is_train=True):
        super().__init__()
        ds = pd.read_csv(csv_fn)
        self.data = ds.to_numpy()
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            img = self.data[index, 1:]
            img = img.reshape((1, 28, 28))
            return img, self.data[index, 0]
        else:
            img = self.data[index, :]
            img = img.reshape((1, 28, 28))
            return img, index
    
    def __len__(self):
        return self.data.shape[0]
    

def split_dataset(dataset, train_percent, train_batch_size, test_batch_size):
    n_train = int(len(dataset) * train_percent)
    n_test = len(dataset) - n_train
    train, test = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def metrics(x, y):
    x = torch.squeeze(x)
    y = torch.squeeze(y)
    if x.shape:
        oa = float(torch.sum(x==y) / len(x))
        return oa
    else:
        return None
    

def train(train_csv, dst_pth_dir):
    device = torch.device('mps')
    # device = torch.device('cpu')
    lr = 1e-4
    epochs = 50
    batch_size = 100
    net = AlexNet(input_channels=1, output_channels=10)
    net.to(device=device)
    dataset = MnistDataset(train_csv, is_train=True)
    train_loader, test_loader = split_dataset(dataset, 0.7, batch_size, batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    oa_best = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        # train dataset
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            labels_predict = net(imgs)
            loss = criterion(labels_predict, labels)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test dataset
        net.eval()
        test_oa = []
        for batch in tqdm(test_loader):
            imgs, labels = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels_pred = net(imgs)
            labels_pred = labels_pred.cpu()
            labels_pred = torch.argmax(labels_pred, axis=1)
            oa = metrics(labels, labels_pred)
            if oa is not None:
                test_oa.append(oa)
        oa_mean = np.mean(test_oa)
        print('epoch %03d: loss=%.2f, test OA=%.2f' % (
            epoch+1, np.mean(epoch_loss), oa_mean
        ))

        # save model
        if epoch >= 3 and oa_mean > oa_best:
            oa_best = oa_mean
            torch.save(net.state_dict(), join(dst_pth_dir, 'E%03d.pth' % (epoch+1)))


def predict(test_csv, net_pth, dst_csv):
    device = torch.device('mps')
    net = AlexNet(input_channels=1, output_channels=10)
    net.to(device=device)
    net.load_state_dict(torch.load(net_pth, map_location=device))
    net.eval()
    dataset = MnistDataset(test_csv, is_train=False)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    labels_pred_stack = []
    for batch in tqdm(loader):
        imgs, index = batch
        imgs = imgs.to(device=device, dtype=torch.float32)
        labels_pred = net(imgs)
        labels_pred = labels_pred.cpu()
        labels_pred = torch.argmax(labels_pred, axis=1)
        for i in range(len(labels_pred)):
            labels_pred_stack.append([int(index[i]), int(labels_pred[i])])
    with open(dst_csv, 'w') as fp:
        fp.write('ImageId,Label\n')
        for item in labels_pred_stack:
            fp.write(f'{item[0]+1},{item[1]}\n')


if __name__ == '__main__':
    train_csv = '/Users/marvin/Documents/kaggle/digit-recognizer/train.csv'
    dst_pth_dir = '/Users/marvin/Documents/kaggle/digit-recognizer/torch_pth/AlexNet/'
    test_csv = '/Users/marvin/Documents/kaggle/digit-recognizer/test.csv'
    net_pth = '/Users/marvin/Documents/kaggle/digit-recognizer/torch_pth/AlexNet/E019.pth'
    dst_csv = '/Users/marvin/Documents/kaggle/digit-recognizer/predict_alexnet.csv'
    # train(train_csv, dst_pth_dir)
    predict(test_csv, net_pth, dst_csv)
