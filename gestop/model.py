'''
Contains the declaration of the neural networks
StaticNet -> A simple FFN to classify static gestures
DynamicNet -> A GRU network which classifies dynamic gestures
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

class StaticNet(LightningModule):
    '''
    The implementation of the model which recognizes static gestures given the hand keypoints.
    '''
    def __init__(self, input_dim, output_classes, gesture_mapping):
        super(StaticNet, self).__init__()

        self.gesture_mapping = gesture_mapping
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x.float())
        return {'loss': F.cross_entropy(output, y.long())}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self(x.float())
        return {'val_step_loss': F.cross_entropy(output, y.long()),
                'val_step_acc': torch.argmax(output, axis=1) == y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        avg_acc = torch.cat([x['val_step_acc'] for x in outputs]).float().mean()
        self.log_dict({'val_loss': avg_loss, 'val_acc': avg_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self(x.float())

        return {'test_acc': torch.argmax(output, axis=1) == y,
                'test_pred': torch.argmax(output, axis=1),
                'test_actual': y}

    def test_epoch_end(self, outputs):
        test_acc = torch.squeeze(torch.cat([x['test_acc'] for x in outputs]).float()).mean()
        test_pred = torch.cat([x['test_pred'] for x in outputs]).cpu().numpy()
        test_actual = torch.cat([x['test_actual'] for x in outputs]).cpu().numpy()

        labels = list(self.gesture_mapping.values())

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

        metrics = {"test_acc":test_acc}

        self.logger.experiment.log({"confusion_matrix":disp.figure_})
        self.logger.log_metrics(metrics)
        self.logger.log_metrics(report)

class DynamicNet(LightningModule):
    '''
    The implementation of the model which recognizes dynamic hand gestures
    given a sequence of keypoints. Consists of a bidirectional GRU connected
    on both sides to a fully conncted layer.
    '''
    def __init__(self, input_dim, output_classes, gesture_mapping):
        super(DynamicNet, self).__init__()

        self.hidden_dim1 = 128
        self.hidden_dim2 = 64
        self.fc1 = nn.Linear(input_dim, self.hidden_dim1)
        self.gru = nn.GRU(input_size=self.hidden_dim1, hidden_size=self.hidden_dim2,
                          bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_dim2*2, output_classes)

        self.time = time.time()
        self.epoch_time = []
        self.gesture_mapping = gesture_mapping

    def replace_last_layer(self, new_output_classes):
        ''' Replacing last layer to learn with new gestures. '''
        self.fc2 = nn.Linear(self.hidden_dim2*2, new_output_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        out, h = self.gru(x)
        # out = pad_packed_sequence(out, batch_first=True)[0]
        last_out = out[:, -1]
        last_out = F.leaky_relu(last_out)
        # last_out = F.leaky_relu(self.fc1(last_out))
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
        self.log('train_loss', avg_loss)

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
        self.log_dict({'val_loss': avg_loss, 'val_acc': avg_acc})

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

        labels = list(self.gesture_mapping.values())

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
