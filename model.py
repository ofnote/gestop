'''
Contains the declaration of the neural network(s)
and their corresponding datasets.
GestureNet -> A simple FFN to classify static gestures
ShrecNet -> A GRU network which classifies dynamic gestures with data from SHREC
'''

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

class ShrecNet(LightningModule):
    '''
    The description of the model which recognizes dynamic hand gestures
    given a sequence of keypoints. Consists of a bidirectional GRU connected
    to a fully conncted layer.
    '''
    def __init__(self, input_dim, output_classes):
        super(ShrecNet, self).__init__()

        self.hidden_dim = 128
        self.gru = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim,
                          bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_classes)
        self.time = time.time()

    def forward(self, x):
        out, h = self.gru(x)
        # out = pad_packed_sequence(out, batch_first=True)[0] #FIXME
        last_out = out[:, -1]
        last_out = F.leaky_relu(last_out)
        last_out = F.leaky_relu(self.fc1(last_out))
        last_out = F.leaky_relu(self.fc2(last_out))
        return last_out

    def training_step(self, batch, batch_idx):
        # x, y, data_len = batch
        x, y = batch

        #FIXME
        # x_packed = pack_padded_sequence(x, data_len, batch_first=True, enforce_sorted=False)
        # output = self(x_packed)
        output = self(x)

        loss = F.cross_entropy(output, y.long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def validation_step(self, batch, batch_idx):
        # x, y, data_len = batch
        x, y = batch

        #FIXME
        # x_packed = pack_padded_sequence(x, data_len, batch_first=True, enforce_sorted=False)
        # output = self(x_packed)
        output = self(x)

        return {'val_loss': F.cross_entropy(output, y.long()),
                'val_acc': np.argmax(output[0].cpu().numpy()) == y}

    def validation_epoch_end(self, outputs):
        epochtime = time.time() - self.time
        self.time = time.time()

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).float().mean()
        tensorboard_logs = {'val_loss': avg_loss, 'epoch_time': epochtime, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self(x)

        return {'test_acc': np.argmax(output[0].cpu().numpy()) == y,
                'test_pred': np.argmax(output[0].cpu().numpy()),
                'test_actual': y}

    def test_epoch_end(self, outputs):
        # print(torch.squeeze(torch.stack([x['test_acc'] for x in outputs]).float()))
        test_acc = torch.squeeze(torch.stack([x['test_acc'] for x in outputs]).float()).mean()
        test_pred = np.array([x['test_pred'] for x in outputs])
        test_actual = torch.squeeze(torch.stack([x['test_actual'] for x in outputs])).cpu().numpy()

        labels=['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation Clockwise', 'Rotation Anticlockwise',
                'Swipe Right', 'Swipe Left', 'Swipe Up', 'Swipe Down', 'Swipe x', 'Swipe +',
                'Swipe V', 'Shake']
        conf_mat = confusion_matrix(test_actual, test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation='vertical')

        plt.show()
        return {'test_acc':test_acc}

#FIXME
def variable_length_collate(batch):
    ''' Custom collate function to handle variable length sequences. '''
    target = torch.empty(len(batch))
    data_lengths = torch.empty(len(batch))
    data = [batch[i][0] for i in range(len(batch))]
    data = pad_sequence(data, batch_first=True)

    for i, (inp, tar) in enumerate(batch):
        data_lengths[i] = inp.shape[0]
        target[i] = tar
    return data, target, data_lengths


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
