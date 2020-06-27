'''
Contains the declaration of the neural network(s) used in the application.
'''

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class GestureNet(nn.Module):
    '''
    The description of the model which recognizes the gestures given the hand keypoints.
    '''
    def __init__(self, input_dim, output_classes):
        super(GestureNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GestureDataset(Dataset):
    ''' Implementation of a GestureDataset which is then loaded into torch's DataLoader'''
    def __init__(self, input_data, target):
        self.input_data = input_data
        self.target = target

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return (self.input_data[index], self.target[index])

class ShrecNet(nn.Module):
    '''
    The description of the model which recognizes dynamic hand gestures given a sequence of keypoints
    '''
    def __init__(self, input_dim, output_classes):
        super(ShrecNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ShrecDataset(Dataset):
    '''
    Implementation of a ShrecDataset which stores the raw SHREC data and formats it as required
    by the network during training.
    '''
    def __init__(self, input_data, target, transform):
        self.input_data = input_data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        x = self.transform(self.input_data[index])
        return (x, self.target[index])
