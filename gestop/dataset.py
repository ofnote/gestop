'''
Contains the declaration of the datasets for the neural networks
StaticDataset -> Dataset of static Gestures for StaticNet
DynamicDataset -> Dataset of SHREC and other dynamic gestures for DynamicNet
'''

from torch.utils.data import Dataset

class StaticDataset(Dataset):
    ''' Implementation of a StaticDataset which is then loaded into torch's DataLoader'''
    def __init__(self, input_data, target):
        self.input_data = input_data
        self.target = target

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return (self.input_data[index], self.target[index])

class DynamicDataset(Dataset):
    '''
    Implementation of a DynamicDataset which stores both SHREC and user data and
    formats it as required by the network during training.
    '''
    def __init__(self, input_data, target, shrec_transform, user_transform):
        self.input_data = input_data
        self.target = target
        self.shrec_transform = shrec_transform
        self.user_transform = user_transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        if self.input_data[index].shape[1] == 44: #SHREC
            x = self.shrec_transform(self.input_data[index])
        else:
            x = self.user_transform(self.input_data[index])
        return (x, self.target[index])
