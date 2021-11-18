'''
Loads the dynamic gesture data from both SHREC and user captured data,
Transforms and prepares dataset in the form of a DataLoader
Trains the network and saves it to disk.
'''

import os
import argparse
import logging
from functools import partial
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from ..model import DynamicNet, init_weights, variable_length_collate
from ..dataset import DynamicDataset
from ..config import Config, get_seed, package_directory
from ..util.utils import calc_polar, init_seed

def normalize(seq):
    '''
    Transformation function. Takes in keypoints and normalizes them.
    Sequence of shape (1,seq_length,44) where the first index is batchsize (set to 1)
    '''
    norm = torch.empty(seq[0].shape)
    i = 0
    for frame in seq[0]: # iterate over each frame in a sequence
        odd = torch.tensor([frame[x] if x%2 == 0 else 0 for x in range(len(frame))])/640
        even = torch.tensor([frame[x] if x%2 != 0 else 0 for x in range(len(frame))])/480

        # Clipping values at zero and one.
        odd[odd < 0] = 0
        odd[odd > 1] = 1

        even[even < 0] = 0
        even[even > 1] = 1

        norm[i] = odd + even
        i += 1
    return norm

def smooth(seq):
    '''
    Transformation Function. Performs exponential smoothing on sequence with factor 'alpha'
    '''
    alpha = 0.9
    smoothed = torch.empty(seq.shape)
    i=0
    for s in seq:
        j=0
        last = s[0] #First timestep
        for point in s:
            smoothval = last*alpha + (1-alpha)*point
            smoothed[i][j] = smoothval
            last = smoothval
            j+=1
        i+=1
    return smoothed


def resample_and_jitter(seq):
    '''
    Transformation Function. Augments the data in the following ways:
    Randomly resamples the sequences to a different length. Adds noise
    to training data to make a more robust network.
    '''
    # Probability of transformation
    p_jitter = p_resample = 0.6
    resampled_len = list(np.arange(0.5, 1.0, 0.05))
    p_sample_len = np.random.choice(resampled_len)

    if np.random.random() < p_resample:
        sample = np.random.choice(a=[True, False], size=(len(seq)),
                                  p=[p_sample_len, 1 - p_sample_len])
        seq = seq[torch.from_numpy(sample)]

    if np.random.random() < p_jitter:
        noise = np.random.normal(size=np.array(seq.shape), scale=0.05)
        seq += noise

    return seq.float()


def format_shrec(C, seq):
    '''
    Transformation Function. Formats the SHREC data as per mediapipe output.
    '''
    tmp_seq = torch.zeros((len(seq),42))
    # Make a new sequence without Palm keypoint
    for i, iseq in enumerate(seq):
        tmp_seq[i] = torch.cat([iseq[0:2], iseq[4:]])
    seq = tmp_seq
    return construct_seq(C, seq)

def format_user(C, seq):
    '''
    Transformation Function. Formats the user data as per mediapipe output.
    '''
    tmp_seq = torch.zeros((len(seq), 42))
    for i, iseq in enumerate(seq):
        # Remove Z-axis coordinates
        all_index = set(range(63))
        del_index = set([i-1 for i in range(64) if i%3==0])
        keep_index = all_index - del_index
        count = 0
        for k in keep_index:
            tmp_seq[i][count] = iseq[k]
            count+=1
    seq = tmp_seq
    return construct_seq(C, seq)

def construct_seq(C, seq):
    '''
    Constructs the final sequence for the transformed data.
    '''
    new_seq = torch.zeros((len(seq),C.dynamic_input_dim))
    for i, iseq in enumerate(seq):
        # Absolute coords
        new_seq[i][0] = iseq[0]
        new_seq[i][1] = iseq[1]

        # Time diff coords
        if i == 0: #start of sequence
            new_seq[i][2] = 0
            new_seq[i][3] = 0
        else:
             x = iseq[0] - new_seq[i-1][0]
             y = iseq[1] - new_seq[i-1][1]
             new_seq[i][2], new_seq[i][3] = calc_polar(x, y)

        for j in range(4):
            # calculate L01, L12, L23, L34
            x = iseq[2*j+2] - iseq[2*j] #L__X
            y = iseq[2*j+3] - iseq[2*j+1] #L__Y
            new_seq[i][4+2*j], new_seq[i][4+2*j+1] = x, y

        for j in range(3):
            # calculate L56, L67, L78
            x = iseq[2*j+12] - iseq[2*j+10]
            y = iseq[2*j+13] - iseq[2*j+11]
            new_seq[i][12+2*j], new_seq[i][12+2*j+1] = x, y

            # calculate L910, L1011, L1112
            x = iseq[2*j+20] - iseq[2*j+18]
            y = iseq[2*j+21] - iseq[2*j+19]
            new_seq[i][18+2*j], new_seq[i][18+2*j+1] = x, y

            # calculate L1314, L1415, L1516
            x = iseq[2*j+28] - iseq[2*j+26]
            y = iseq[2*j+29] - iseq[2*j+27]
            new_seq[i][24+2*j], new_seq[i][24+2*j+1] = x, y

            # calculate L1718, L1819, L1920
            x = iseq[2*j+36] - iseq[2*j+34]
            y = iseq[2*j+37] - iseq[2*j+35]
            new_seq[i][30+2*j], new_seq[i][30+2*j+1] = x, y

    return new_seq


def read_shrec_data(base_directory):
    ''' Reads data from SHREC2017 dataset files. '''

    gesture_dir = ['gesture_'+str(i) for i in range(1, 15)]
    gesture_arr = []
    target_arr = []
    gesture_no = 0

    for gesture in [os.path.join(base_directory, i) for i in gesture_dir]: # for each gesture
        for finger in ['/finger_1', '/finger_2']:
            for subject in os.listdir(gesture+finger):
                for essai in os.listdir(gesture+finger+'/'+subject):
                    data = np.loadtxt(gesture+finger+'/'+subject+'/'+essai+'/skeletons_image.txt')
                    gesture_arr.append(data)
                    target_arr.append(gesture_no)
        gesture_no += 1

    with open(os.path.join(package_directory, 'data/shrec_gesture_mapping.json'), 'r') as jsonfile:
        shrec_dict = json.load(jsonfile)

    return gesture_arr, target_arr, shrec_dict

def read_user_data(base_directory):
    ''' Reads the user collected data. '''

    gesture_arr = []
    target_arr = []
    gesture_no = 14 #no. of gestures in SHREC
    user_dict = {}

    if base_directory is None:
        return gesture_arr, target_arr, user_dict

    for gesture in os.listdir(base_directory): # for each gesture
        for g in os.listdir(base_directory+'/'+gesture):
            data = np.loadtxt(base_directory+'/'+gesture+'/'+g)
            gesture_arr.append(data)
            target_arr.append(gesture_no)
        user_dict[gesture_no] = gesture
        gesture_no += 1

    return gesture_arr, target_arr, user_dict

def read_data(seed_val, shrec_directory, user_directory):
    ''' Read both user data and SHREC data. '''
    gesture_shrec, target_shrec, shrec_dict = read_shrec_data(shrec_directory)
    gesture_user, target_user, user_dict = read_user_data(user_directory)

    shrec_dict.update(user_dict)

    gesture = gesture_shrec + gesture_user
    target = target_shrec + target_user

    train_x, test_x, train_y, test_y = train_test_split(gesture, target, test_size=0.2,
                                                        random_state=seed_val)
    return train_x, test_x, train_y, test_y, shrec_dict

def choose_collate(collate_fn, C):
    ''' Returns None(default collate) if batch size is 1, else custom collate. '''
    if C.dynamic_batch_size == 1:
        return None
    return collate_fn

def main():
    ''' Main '''

    parser = argparse.ArgumentParser(description='A program to train a neural network \
    to recognize dynamic hand gestures.')
    parser.add_argument("--exp-name", help="The name with which to log the run.", type=str)
    parser.add_argument("--shrec-directory", help="The directory of SHREC.", required=True)
    parser.add_argument("--user-directory", help="The directory in which user collected gesture data is stored.")
    parser.add_argument("--use-pretrained", help="Use pretrained model.",
                        dest="pretrained", action="store_true")

    args = parser.parse_args()
    init_seed(get_seed())

    ##################
    # INPUT PIPELINE #
    ##################

    with open(os.path.join(package_directory, 'data/dynamic_gesture_mapping.json'), 'r') as f:
        old_gesture_mapping = json.load(f) # Keep old mapping in case we pretrain
        old_output_classes = len(old_gesture_mapping)
    train_x, test_x, train_y, test_y, gesture_mapping = read_data(get_seed(), args.shrec_directory, args.user_directory)
    logging.info(gesture_mapping)
    with open(os.path.join(package_directory, 'data/dynamic_gesture_mapping.json'), 'w') as f:
        f.write(json.dumps(gesture_mapping)) # Store new mapping

    C = Config(lite=True)

    # Higher order function to pass configuration as argument
    shrec_to_mediapipe = partial(format_shrec, C)
    user_to_mediapipe = partial(format_user, C)

    # Custom transforms to prepare data.
    shrec_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Lambda(resample_and_jitter),
        transforms.Lambda(shrec_to_mediapipe),
    ])
    user_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.squeeze),
        transforms.Lambda(resample_and_jitter),
        transforms.Lambda(user_to_mediapipe),
    ])

    train_loader = DataLoader(DynamicDataset(train_x, train_y, shrec_transform, user_transform),
                              num_workers=10, batch_size=C.dynamic_batch_size,
                              collate_fn=choose_collate(variable_length_collate, C))
    val_loader = DataLoader(DynamicDataset(test_x, test_y, shrec_transform, user_transform),
                            num_workers=10, batch_size=C.dynamic_batch_size,
                            collate_fn=choose_collate(variable_length_collate, C))

    ############
    # TRAINING #
    ############

    # Use pretrained model
    if args.pretrained:
        model = DynamicNet(C.dynamic_input_dim, old_output_classes, old_gesture_mapping)
        model.load_state_dict(torch.load(C.dynamic_path))
        model.replace_last_layer(C.dynamic_output_classes)
        model.gesture_mapping = gesture_mapping
    else:
        model = DynamicNet(C.dynamic_input_dim, C.dynamic_output_classes, gesture_mapping)
        model.apply(init_weights)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
    )

    # No name is given as a command line flag.
    if args.exp_name is None:
        args.exp_name = "dynamic_net"

    wandb_logger = pl_loggers.WandbLogger(save_dir=os.path.join(package_directory, 'logs/'),
                                          name=args.exp_name,
                                          project='gestop')

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      logger=wandb_logger,
                      min_epochs=20,
                      accumulate_grad_batches=C.grad_accum,
                      callbacks=[early_stopping])

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), C.dynamic_path)

    trainer.test(model, test_dataloaders=val_loader)


if __name__ == '__main__':
    main()
