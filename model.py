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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

class GestureNet(LightningModule):
    '''
    The implementation of the model which recognizes static gestures given the hand keypoints.
    '''
    def __init__(self, input_dim, output_classes):
        super(GestureNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x.float())

        loss = F.cross_entropy(output, y.long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self(x.float())

        return {'val_loss': F.cross_entropy(output, y.long()),
                'val_acc': np.argmax(output[0].cpu().numpy()) == y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs[:-1]]).float().mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self(x)

        return {'test_acc': np.argmax(output[0].cpu().numpy()) == y}

    def test_epoch_end(self, outputs):
        test_acc = torch.squeeze(torch.stack([x['test_acc'] for x in outputs]).float()).mean()
        return {'test_acc':test_acc}

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
    The implementation of the model which recognizes dynamic hand gestures
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
        self.epoch_time = []

    def replace_layers(self, new_output_classes):
        ''' Replacing last layer to learn with new gestures. '''
        self.fc2 = nn.Linear(self.hidden_dim, new_output_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out, h = self.gru(x)
        # out = pad_packed_sequence(out, batch_first=True)[0]
        last_out = out[:, -1]
        last_out = F.leaky_relu(last_out)
        last_out = F.leaky_relu(self.fc1(last_out))
        last_out = F.leaky_relu(self.fc2(last_out))
        return last_out

    def training_step(self, batch, batch_idx):
        # x, y, data_len = batch
        x, y = batch

        # x_packed = pack_padded_sequence(x, data_len, batch_first=True, enforce_sorted=False)
        # output = self(x_packed)
        output = self(x)

        loss = F.cross_entropy(output, y.long())
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log':tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # x, y, data_len = batch
        x, y = batch

        # x_packed = pack_padded_sequence(x, data_len, batch_first=True, enforce_sorted=False)
        # output = self(x_packed)
        output = self(x)

        return {'val_loss': F.cross_entropy(output, y.long()),
                'val_acc': torch.argmax(output, axis=1) == y}

    def validation_epoch_end(self, outputs):
        self.epoch_time.append(time.time() - self.time)
        self.time = time.time()

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.cat([x['val_acc'] for x in outputs]).float().mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # x, y, data_len = batch
        x, y = batch

        # x_packed = pack_padded_sequence(x, data_len, batch_first=True, enforce_sorted=False)
        # output = self(x_packed)
        output = self(x)

        return {'test_acc': torch.argmax(output, axis=1) == y,
                'test_pred': torch.argmax(output, axis=1),
                'test_actual': y}

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([x['test_acc'] for x in outputs]).float().mean()
        test_pred = torch.cat([x['test_pred'] for x in outputs]).cpu().numpy()
        test_actual = torch.cat([x['test_actual'] for x in outputs]).cpu().numpy()

        labels = ['Grab', 'Tap', 'Expand', 'Pinch', 'RClockwise', 'RCounterclockwise',
                  'Swipe Right', 'Swipe Left', 'Swipe Up', 'Swipe Down', 'Swipe x', 'Swipe +',
                  'Swipe V', 'Shake', 'Circle']

        report = classification_report(test_actual, test_pred,
                                       target_names=labels, output_dict=True)
        # String representation for easy vieweing
        str_report = classification_report(test_actual, test_pred, target_names=labels)
        print(str_report)

        # Format the report
        report.pop('accuracy')
        report.pop('macro avg')
        for key, value in report.items():
            report[key].pop('support')

        conf_mat = confusion_matrix(test_actual, test_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues,
                         ax=None, xticks_rotation='vertical')
        disp.figure_.set_size_inches(12, 12)

        avg_epoch_time = sum(self.epoch_time)/max(len(self.epoch_time), 1)

        metrics = {"test_acc":test_acc, "average_epoch_time":avg_epoch_time}

        self.logger.experiment.log({"confusion_matrix":disp.figure_})
        self.logger.log_metrics(metrics)
        self.logger.log_metrics(report)

        return metrics

def init_weights(m):
    ''' Initializes weights of network with Xavier Initialization.'''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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
    Implementation of a ShrecDataset which stores both SHREC and user data and
    formats it as required by the network during training.
    '''
    def __init__(self, input_data, target, base_transform, final_transform):
        self.input_data = input_data
        self.target = target
        self.base_transform = base_transform
        self.final_transform = final_transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        x = self.base_transform(self.input_data[index])
        l = len(x[0])
        if l == 63: # user data (21*3=63)
            x = self.final_transform[1](x)
        else: # shrec data (22*2=44)
            x = self.final_transform[0](x)
        return (x, self.target[index])
