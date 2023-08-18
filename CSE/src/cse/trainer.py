"""Trainer Module to assist with Training.
"""
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
from time import time
from typing import Callable, Tuple, List
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from math import sqrt, ceil, floor
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18

class Trainer:
    """Trainer Class
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        loss_func: Callable,  # from torch.nn.functional.*
        optimizer: torch.optim.Optimizer,
        max_run_time: float,
        snapshot_name: str,
    ) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        print(f"Model loaded on device: {next(model.parameters()).device}")
        self.train_data = train_data
        self.valid_data = valid_data
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch=-1, verbose=False)
        # Hours to seconds, training will stop at this time
        self.max_run_time = max_run_time * 60**2
        self.save_path = "training_saves/" + snapshot_name
        self.epochs_run = 0  # current epoch tracker
        self.run_time = 0.0  # current run_time tracker
        self.train_loss_history = list()
        self.valid_loss_history = list()
        self.epoch_times = list()
        self.lowest_loss = np.Inf
        self.train_loss = np.Inf
        self.valid_loss = np.Inf
        # Loading in existing training session if the save destination already exists
        if os.path.exists(self.save_path):
            print("Loading snapshot")
            self._load_snapshot(self.save_path)
        if self.train_loss_history:
            self.train_loss = self.train_loss_history[-1]
            self.valid_loss = self.valid_loss_history[-1]
            
    def _load_snapshot(self, snapshot_path):
        loc = "cuda:0"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.run_time = snapshot['RUN_TIME']
        self.train_loss_history = snapshot['TRAIN_HISTORY']
        self.valid_loss_history = snapshot['VALID_HISTORY']
        self.epoch_times = snapshot['EPOCH_TIMES']
        self.lowest_loss = snapshot['LOWEST_LOSS']
        print(f"Resuming training from save at Epoch {self.epochs_run}")

    def _calc_validation_loss(self, source, targets) -> float:
        self.model.eval()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        self.model.train()
        return float(loss.item())

    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _run_epoch(self):
        b_sz = len(next(iter(self.train_data))[0])
        if self.epochs_run % 10 == 0:
            print(
                f"\nEpoch: {self.epochs_run} | Batch_SZ: {b_sz} ", end="")
            print(
                f"| Steps: {len(self.train_data)} ", end="")
            print(
                f"| T_loss: {self.train_loss:.3f} | V_loss: {self.valid_loss:.3f}")
        # self.train_data.sampler.set_epoch(self.epochs_run)
        train_loss = 0
        valid_loss = 0

        # Train Loop
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            train_loss += self._run_batch(source, targets)

        # Calculating Validation loss
        for source, targets in self.valid_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            valid_loss += self._calc_validation_loss(source, targets)

        # Update loss history & scheduler.step
        self.scheduler.step()
        self.train_loss_history.append(train_loss/len(self.train_data))
        self.valid_loss_history.append(valid_loss/len(self.valid_data))
        self.train_loss, self.valid_loss = self.train_loss_history[-1], self.valid_loss_history[-1]

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": self.epochs_run,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY": self.train_loss_history,
            "VALID_HISTORY": self.valid_loss_history,
            "EPOCH_TIMES": self.epoch_times,
            "LOWEST_LOSS": self.lowest_loss
        }
        torch.save(snapshot, self.save_path)
        print(f"Training snapshot saved after Epoch: {self.epochs_run} | save_name: {self.save_path}")

    def train(self):
        for _ in range(self.epochs_run, self.epochs_run + 1000):
            start = time()
            self._run_epoch()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times.append(elapsed_time)
            start = time()
            self.epochs_run += 1
            if self.valid_loss_history[-1] < self.lowest_loss:
                self.lowest_loss = self.valid_loss_history[-1]
                self._save_snapshot()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times[-1] += elapsed_time
            if self.epochs_run % 10 == 1:
                print(
                    f'Current Train Time: {self.run_time//60**2} hours & {((self.run_time%60.0**2)/60.0):.2f} minutes')
            if (self.run_time > self.max_run_time):
                print(
                    f"Training completed -> Total train time: {self.run_time:.2f} seconds")
                break

        # Saving import metrics to analyze training on local machine
        train_metrics = {
                "EPOCHS_RUN": self.epochs_run,
                "RUN_TIME": self.run_time,
                "TRAIN_HISTORY": self.train_loss_history,
                "VALID_HISTORY": self.valid_loss_history,
                "EPOCH_TIMES": self.epoch_times,
                "LOWEST_LOSS": self.lowest_loss
        }
        torch.save(train_metrics, self.save_path[:-3] + "_metrics.pt")
            

class MNIST_model(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(MNIST_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1  , out_channels=32 , kernel_size=3, stride=1), nn.BatchNorm2d(32) , nn.ReLU(),
            nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=3, stride=1), nn.BatchNorm2d(48) , nn.ReLU(),
            nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=3, stride=1), nn.BatchNorm2d(64) , nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels=80 , kernel_size=3, stride=1), nn.BatchNorm2d(80) , nn.ReLU(),
            nn.Conv2d(in_channels=80 , out_channels=96 , kernel_size=3, stride=1), nn.BatchNorm2d(96) , nn.ReLU(),
            nn.Conv2d(in_channels=96 , out_channels=112, kernel_size=3, stride=1), nn.BatchNorm2d(112), nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3, stride=1), nn.BatchNorm2d(144), nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3, stride=1), nn.BatchNorm2d(160), nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3, stride=1), nn.BatchNorm2d(176), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11264, n_classes),
            nn.BatchNorm1d(n_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        out = self.final_conv(x)
        return self.classifier(out)
    
# class MNIST_model(torch.nn.Module):
#     def __init__(self, n_classes=10):
#         super(MNIST_model, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels=1  , out_channels=32 , kernel_size=3, stride=1), nn.BatchNorm2d(32) , nn.ReLU(),
#             nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=3, stride=1), nn.BatchNorm2d(48) , nn.ReLU(),
#             nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=3, stride=1), nn.BatchNorm2d(64) , nn.ReLU(),
#             nn.Conv2d(in_channels=64 , out_channels=80 , kernel_size=3, stride=1), nn.BatchNorm2d(80) , nn.ReLU(),
#             nn.Conv2d(in_channels=80 , out_channels=96 , kernel_size=3, stride=1), nn.BatchNorm2d(96) , nn.ReLU(),
#             nn.Conv2d(in_channels=96 , out_channels=112, kernel_size=3, stride=1), nn.BatchNorm2d(112), nn.ReLU(),
#             nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3, stride=1), nn.BatchNorm2d(144), nn.ReLU(),
#             nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3, stride=1), nn.BatchNorm2d(160), nn.ReLU(),
#             nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3, stride=1), nn.BatchNorm2d(176), nn.ReLU()
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(11264, n_classes),
#             nn.BatchNorm1d(n_classes)
#         )
        
#     def forward(self, x):
#         x = self.encoder(x)
#         return self.classifier(x)

    
class MNIST_model_modified(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(MNIST_model_modified, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1  , out_channels=32 , kernel_size=3, stride=1), nn.BatchNorm2d(32) , nn.ReLU(),
            nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=3, stride=1), nn.BatchNorm2d(48) , nn.ReLU(),
            nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=3, stride=1), nn.BatchNorm2d(64) , nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels=80 , kernel_size=3, stride=1), nn.BatchNorm2d(80) , nn.ReLU(),
            nn.Conv2d(in_channels=80 , out_channels=96 , kernel_size=3, stride=1), nn.BatchNorm2d(96) , nn.ReLU(),
            nn.Conv2d(in_channels=96 , out_channels=112, kernel_size=3, stride=1), nn.BatchNorm2d(112), nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3, stride=1), nn.BatchNorm2d(144), nn.ReLU(),
            nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3, stride=1), nn.BatchNorm2d(160), nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3, stride=1), nn.BatchNorm2d(176), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11264, n_classes),
            nn.BatchNorm1d(n_classes)
        )
        
        
class gradcam_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(6400, 128, bias=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10, bias=True)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return x

    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
def resnet_model_modified():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.fc = nn.Linear(in_features=64, out_features=10, bias=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    model.layer1 = nn.Sequential(
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer2 = Identity()
    model.layer3 = Identity()
    model.layer4 = Identity()
    return model

    
class Custom_MNIST_Dataset(Dataset):
    def __init__(self, sliced_dataset: List[Tuple[torch.Tensor, int]]):
        self.sliced_dataset = sliced_dataset
        
    def __len__(self):
        return len(self.sliced_dataset)
    
    def __getitem__(self, index):
        return self.sliced_dataset[index]
    
            
def create_train_objs(learning_rate: float = 0.001) -> Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]:
    """Used to instantiate 3 training objects. Model, loss_func, and Optimizer
    Returns:
        Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]: 
        tuple of model, Loss Function, and Optimizer
    """
    model = gradcam_model()
    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_func, optimizer
    
    
def create_dataloaders_MNIST(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Used to instantiate 2 Dataloaders training.
    Args:
        batch_size (int): batch size of each device
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: tuple of training, validation, and test dataloaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # transforms.RandomAffine(20, translate=(0.20,0.20))
    ])

    train_dataset = torchvision.datasets.MNIST('.', train=True, transform=transform)
    
    train_split = ceil(len(train_dataset) * 0.90)
    valid_split = floor(len(train_dataset) * 0.10)
    
    test_data = torchvision.datasets.MNIST('.', train=False, transform=transform)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    train_data, valid_data = random_split(
        train_dataset, [train_split, valid_split], generator=generator)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,  # Allocates samples into page-locked memory, speeds up data transfer to GPU
        shuffle=True,  
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader


def create_dataloaders_MNIST_fashion(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Used to instantiate 2 Dataloaders training.
    Args:
        batch_size (int): batch size of each device
        allowed_classes List[int]: allowed classes
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: tuple of training, validation, and test dataloaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.RandomAffine(20, translate=(0.20,0.20))
    ])

    train_dataset = torchvision.datasets.FashionMNIST('.', train=True, transform=transform)
    
    train_split = ceil(len(train_dataset) * 0.90)
    valid_split = floor(len(train_dataset) * 0.10)
    
    test_data = torchvision.datasets.FashionMNIST('.', train=False, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(42)
    train_data, valid_data = random_split(
        train_dataset, [train_split, valid_split], generator=generator)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,  # Allocates samples into page-locked memory, speeds up data transfer to GPU
        shuffle=True,  
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader


def print_model_test_stats(model: torch.nn.Module,
                           test_loader: DataLoader,
                           labels_map: dict,
                           loss_func: Callable = F.cross_entropy
                           ) -> None:
    # initialize lists to monitor test loss and accuracy
    classes_num = 10
    test_loss = 0.0
    class_correct = list(0. for i in range(classes_num))
    class_total = list(0. for i in range(classes_num))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(len(target)):
            label = target[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    t_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(t_loss))

    for i in range(classes_num):
        if class_total[i] > 0:
            percent_correct = 100 * class_correct[i] / class_total[i]
            print(
                f'Test Accuracy of Class: {labels_map[i]:19s}: {percent_correct:3.2f}%',
                f'({np.sum(int(class_correct[i]))}/{np.sum(int(class_total[i]))})')
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (labels_map[i]))

    percent_correct = 100.0 * np.sum(class_correct) / np.sum(class_total)
    print(
        f'\nTest Accuracy (Overall): {percent_correct:3.2f}%',
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})')