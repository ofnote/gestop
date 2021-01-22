'''
Describes the implementation of the training procedure for gesture net
'''
import json
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from model import GestureNet, GestureDataset
from config import Config

def init_seed(seed):
    ''' Initializes random seeds for reproducibility '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    loaded_data = DataLoader(GestureDataset(formatted_data, formatted_target),
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

    C = Config(lite=True)
    init_seed(C.seed_val)

    ##################
    # INPUT PIPELINE #
    ##################

    # Read and format the csv
    df = pd.read_csv("gestop/data/static_gestures_data.csv")
    train, test = train_test_split(df, test_size=0.1, random_state=C.seed_val)
    train_X, train_Y = split_dataframe(train)
    test_X, test_Y = split_dataframe(test)

    # One Hot Encoding of the target classes
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)

    le = LabelEncoder()
    le.fit(train_Y)

    # Store encoding to disk
    le_name_mapping = dict(zip([int(i) for i in le.transform(le.classes_)], le.classes_))
    logging.info(le_name_mapping)
    with open('gestop/data/static_gesture_mapping.json', 'w') as f:
        f.write(json.dumps(le_name_mapping))


    train_Y = le.transform(train_Y)
    test_Y = le.transform(test_Y)

    train_loader = format_and_load(train_X, train_Y, C.static_batch_size)
    test_loader = format_and_load(test_X, test_Y, C.static_batch_size)

    gesture_net = GestureNet(C.static_input_dim, C.static_output_classes, C.static_gesture_mapping)

    early_stopping = EarlyStopping(
        patience=3,
        verbose=True,
    )

    wandb_logger = pl_loggers.WandbLogger(save_dir='gestop/logs/',
                                          name='gesture_net',
                                          project='gestop')

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      logger=wandb_logger,
                      min_epochs=C.min_epochs,
                      early_stop_callback=early_stopping)
    trainer.fit(gesture_net, train_loader, test_loader)
    # gesture_net.load_state_dict(torch.load(PATH))
    trainer.test(gesture_net, test_dataloaders=test_loader)

    ################
    # SAVING MODEL #
    ################

    torch.save(gesture_net.state_dict(), C.static_path)


if __name__ == '__main__':
    main()
