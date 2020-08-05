'''
Loads the dynamic gesture data from both SHREC and user captured data,
Transforms and prepares dataset in the form of a DataLoader
Trains the network and saves it to disk.
'''

import os
import argparse
import math
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
from model import ShrecNet, ShrecDataset, init_weights, variable_length_collate
from config import Config

def init_seed(seed):
    ''' Initializes random seeds for reproducibility '''
    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def calc_polar(x,y):
    return (x**2 + y**2)**0.5, math.atan2(y, x)/math.pi

def format_mediapipe(C, seq):
    '''
    Transformation Function. Formats the normalized keypoints as per mediapipe output.
    '''
    if len(seq[0]) == 63: # User collected data
        seq = reshape_user(seq)
    else: # SHREC data
        # Make a new sequence without Palm keypoint
        tmp_seq = torch.zeros((len(seq),42))
        for i, iseq in enumerate(seq):
            tmp_seq[i] = torch.cat([iseq[0:2], iseq[4:]])
        seq = tmp_seq
    new_seq = torch.zeros((len(seq),C.dynamic_input_dim))
    for i, iseq in enumerate(seq):
        # Absolute
        new_seq[i][0] = iseq[0]
        new_seq[i][1] = iseq[1]

        if i == 0: #start of sequence
            new_seq[i][2] = 0
            new_seq[i][3] = 0
        else: # Time diff coordinate
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

def reshape_user(seq):
    ''' Reshapes user collected data into the same shape as SHREC data. '''
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
    return tmp_seq

def read_shrec_data():
    ''' Reads data from SHREC2017 dataset files. '''
    # Change this as per your system
    base_directory = "/home/sriramsk/Desktop/HandGestureDataset_SHREC2017"

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

    with open('data/shrec_gesture_mapping.json', 'r') as jsonfile:
        shrec_dict = json.load(jsonfile)

    return gesture_arr, target_arr, shrec_dict

def read_user_data():
    ''' Reads the user collected data. '''
    base_directory = "data/dynamic_gestures"

    gesture_arr = []
    target_arr = []
    gesture_no = 14 #no. of gestures in SHREC
    user_dict = {}

    for gesture in os.listdir(base_directory): # for each gesture
        for g in os.listdir(base_directory+'/'+gesture):
            data = np.loadtxt(base_directory+'/'+gesture+'/'+g)
            gesture_arr.append(data)
            target_arr.append(gesture_no)
        user_dict[gesture_no] = gesture
        gesture_no += 1

    return gesture_arr, target_arr, user_dict

def read_data(seed_val):
    ''' Read both user data and SHREC data. '''
    gesture_shrec, target_shrec, shrec_dict = read_shrec_data()
    gesture_user, target_user, user_dict = read_user_data()

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

    args = parser.parse_args()

    C = Config(lite=True, pretrained=False)
    init_seed(C.seed_val)

    ##################
    # INPUT PIPELINE #
    ##################

    train_x, test_x, train_y, test_y, gesture_mapping = read_data(C.seed_val)
    with open('data/dynamic_gesture_mapping.json', 'w') as f:
        f.write(json.dumps(gesture_mapping))

    # Higher order function to pass configuration to format_mediapipe
    format_med = partial(format_mediapipe, C)

    # Custom transforms to prepare data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Lambda(resample_and_jitter),
        transforms.Lambda(format_med),
    ])

    train_loader = DataLoader(ShrecDataset(train_x, train_y, transform),
                              num_workers=10, batch_size=C.dynamic_batch_size,
                              collate_fn=choose_collate(variable_length_collate, C))
    val_loader = DataLoader(ShrecDataset(test_x, test_y, transform),
                            num_workers=10, batch_size=C.dynamic_batch_size,
                            collate_fn=choose_collate(variable_length_collate, C))

    ############
    # TRAINING #
    ############

    # Use pretrained SHREC model
    if C.pretrained:
        model = ShrecNet(C.dynamic_input_dim, C.shrec_output_classes, gesture_mapping)
        model.load_state_dict(torch.load(C.shrec_path))
        model.replace_layers(C.dynamic_output_classes)
    else:
        model = ShrecNet(C.dynamic_input_dim, C.dynamic_output_classes, gesture_mapping)
        model.apply(init_weights)

    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
    )

    # No name is given as a command line flag.
    if args.exp_name is None:
        args.exp_name = "default"

    wandb_logger = pl_loggers.WandbLogger(save_dir='logs/',
                                          name=args.exp_name,
                                          project='gestures-mediapipe')

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      logger=wandb_logger,
                      min_epochs=20,
                      accumulate_grad_batches=C.grad_accum,
                      early_stop_callback=early_stopping)

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), C.dynamic_path)

    trainer.test(model, test_dataloaders=val_loader)


if __name__ == '__main__':
    main()
