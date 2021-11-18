'''
Describes the implementation of the training procedure for gesture net
'''
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from ..model import StaticNet
from ..dataset import StaticDataset
from ..config import Config, get_seed, package_directory
from ..util.utils import update_static_mapping, init_seed

def split_dataframe(data):
    ''' Splits the given dataset into input and target data. '''
    input_data = data.drop('GESTURE', axis=1)
    target_data = data['GESTURE']

    return input_data, target_data


def format_and_load(dataset, target, batchsize):
    '''
    Formats the input dataset as expected by the network and returns a DataLoader.
    '''
    formatted_data = torch.tensor(make_vector(dataset))
    formatted_target = torch.tensor(target)

    loaded_data = DataLoader(StaticDataset(formatted_data, formatted_target),
                             batch_size=batchsize, num_workers=10)
    return loaded_data

def make_vector(dataset):
    '''
    Formats the input in the following fashion:
    Calculates a 3D vector between each adjacent keypoint
    Some vectors are not calculated as they are useless. E.g. LM0 <-> LM5
    Each vector consists of 3 datapoints (x, y, z)
    Also consists of which hand is used for the gesture.
    '''
    row_length = 49
    formatted = np.empty((len(dataset), row_length))

    i = 0
    for index, row in dataset.iterrows():
        #pd.set_option('display.max_rows', None)
        #print(dataset.loc[i])
        for j in range(4):
            # calculate L01, L12, L23, L34
            formatted[i][3*j] = row[3*j+3] - row[3*j] #L__X
            formatted[i][3*j+1] = row[3*j+4] - row[3*j+1] #L__Y
            formatted[i][3*j+2] = row[3*j+5] - row[3*j+2] #L__Z
            # formatted[i][3*j+2] = 0

        for j in range(3):
            # calculate L56, L67, L78
            formatted[i][3*j+12] = row[3*j+18] - row[3*j+15]
            formatted[i][3*j+13] = row[3*j+19] - row[3*j+16]
            formatted[i][3*j+14] = row[3*j+20] - row[3*j+17]
            # formatted[i][3*j+14] = 0

            # calculate L910, L1011, L1112
            formatted[i][3*j+21] = row[3*j+30] - row[3*j+27]
            formatted[i][3*j+22] = row[3*j+31] - row[3*j+28]
            formatted[i][3*j+23] = row[3*j+32] - row[3*j+29]
            # formatted[i][3*j+23] = 0

            # calculate L1314, L1415, L1516
            formatted[i][3*j+30] = row[3*j+42] - row[3*j+39]
            formatted[i][3*j+31] = row[3*j+43] - row[3*j+40]
            formatted[i][3*j+32] = row[3*j+44] - row[3*j+41]
            # formatted[i][3*j+32] = 0

            # calculate L1718, L1819, L1920
            formatted[i][3*j+39] = row[3*j+54] - row[3*j+51]
            formatted[i][3*j+40] = row[3*j+55] - row[3*j+52]
            formatted[i][3*j+41] = row[3*j+56] - row[3*j+53]
            # formatted[i][3*j+41] = 0

        formatted[i][48] = row['HAND']
        i += 1
    return formatted

def calc_accuracy(ans, pred):
    ''' Calcaulates the accuracy of the predictions made in a batch'''
    pred = np.argmax(pred, axis=1).flatten()
    return np.sum(np.equal(pred, ans)) / len(ans)

def main():
    ''' Main '''

    parser = argparse.ArgumentParser(description='A program to train a neural network to detect static hand gestures.')
    parser.add_argument("--static-gesture-filepath", help="Path to the file containing static gestures.", required=True)

    args = parser.parse_args()

    encoder = update_static_mapping(args.static_gesture_filepath) # Update the static mapping before initializing Config
    C = Config(lite=True)
    init_seed(get_seed())

    ##################
    # INPUT PIPELINE #
    ##################

    # Read and format the csv
    df = pd.read_csv(args.static_gesture_filepath)
    train, test = train_test_split(df, test_size=0.1, random_state=get_seed())
    train_X, train_Y = split_dataframe(train)
    test_X, test_Y = split_dataframe(test)

    # One Hot Encoding of the target classes
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)

    train_Y = encoder.transform(train_Y)
    test_Y = encoder.transform(test_Y)

    train_loader = format_and_load(train_X, train_Y, C.static_batch_size)
    test_loader = format_and_load(test_X, test_Y, C.static_batch_size)

    static_net = StaticNet(C.static_input_dim, C.static_output_classes, C.static_gesture_mapping)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
    )

    wandb_logger = pl_loggers.WandbLogger(save_dir=os.path.join(package_directory, 'logs/'),
                                          name='static_net',
                                          project='gestop')

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      logger=wandb_logger,
                      min_epochs=C.min_epochs,
                      callbacks=[early_stopping])
    trainer.fit(static_net, train_loader, test_loader)
    trainer.test(static_net, test_dataloaders=test_loader)

    ################
    # SAVING MODEL #
    ################

    torch.save(static_net.state_dict(), C.static_path)


if __name__ == '__main__':
    main()
