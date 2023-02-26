''' deep learning
'''
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


class AlexNet(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
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

    def __getitem__(self, index):
        img = self.data[index, 1:]
        img = img.reshape((1, 28, 28))
        return img, self.data[index, 0]
    
    def __len__(self):
        return self.data.shape[0]
    

def split_dataset(dataset, train_percent, train_batch_size, test_batch_size):
    n_train = int(len(dataset) * train_percent)
    n_test = len(dataset) - n_train
    train, test = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, test_loader
    

def train(train_csv):
    device = torch.device('mps')
    # device = torch.device('cpu')
    lr = 1e-4
    epochs = 100
    batch_size = 50
    net = AlexNet(input_channels=1, output_channels=10)
    net.to(device=device)
    dataset = MnistDataset(train_csv, is_train=True)
    train_loader, test_loader = split_dataset(dataset, 0.7, batch_size, batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        # train dataset
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            labels_predict = net(imgs)
            loss = criterion(labels_predict, labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test dataset
        test_oa = 0
        for batch in tqdm(test_loader):
            imgs, labels = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels_pred = net(imgs)
            masks_pred = masks_pred.cpu()
            masks_pred = torch.argmax(masks_pred, axis=1)
            masks_pred = torch.squeeze(masks_pred)


if __name__ == '__main__':
    train_csv = '/Users/marvin/Documents/kaggle/digit-recognizer/train.csv'
    train(train_csv)
