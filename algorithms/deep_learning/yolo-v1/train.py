import torch
import torch.nn as nn
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.deep_learning.resunet import data_loader


def main(model_settings, running_params):
    in_channels = model_settings['in_channels']
    out_channels = model_settings['out_channels']
    model_pth = model_settings['model_pth']
    epochs = model_settings['epochs']
    batch_size = model_settings['batch_size']
    lr = model_settings['lr']
    num_workers = model_settings['num_workers']
    # device = torch.device('cpu')
    # device = torch.device('cuda:0')
    device = torch.device('mps')
    checkpoint_dir = running_params['checkpoint_dir']
    img_dir_train = running_params['img_dir_train']
    img_dir_test = running_params['img_dir_test']
    mask_dir_train = running_params['mask_dir_train']
    mask_dir_test = running_params['mask_dir_test']

    net = ResUnet(in_channels, out_channels)
    if model_pth is not None:
        net.load_state_dict(torch.load(model_pth))
    net.to(device=device)
    train_dataset = data_loader.RoadDatasetCUG(img_dir_train, mask_dir_train)
    test_dataset = data_loader.Dataset(img_dir_test, mask_dir_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print('training...')
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            img, mask = batch
            img = img.to(device=device, dtype=torch.float32)
            mask_predict = net(img)
            loss = criterion(mask_predict, mask)
            epoch_loss += loss
            optimizer.zero_grad()  # 将上次backward的梯度清零
            loss.backward()  # 根据loss重新进行梯度反向传播
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()  # 根据梯度更新权重
            if epoch >= 9:
                dst_pth = join(checkpoint_dir, 'epoch_%03d.pth'%(epoch+1))
                torch.save(net.state_dict(), dst_pth)
            
        mean_loss = epoch_loss / len(train_loader)


if __name__ == '__main__':
    model_settings = {
        'model_pth': None,
        'epochs': 100,
        'in_channels': 3,
        'out_channels': 2,
        'lr': 1e-4,
        'batch_size': 5,
        'num_workers': 8,
    }
    running_params = {
        'checkpoint_dir': '',
        'img_dir_train': '',
        'img_dir_test': '',
        'mask_dir_train': '',
        'mask_dir_test': ''
    }
