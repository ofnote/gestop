'''
Loads the dynamic gesture data from the SHREC dataset,
Transforms and prepares dataset in the form of a DataLoader
Trains the network and saves it to disk.
'''

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision import transforms
from model import ShrecNet, ShrecDataset

def init_seed(seed):
    ''' Initializes random seeds for reproducibility '''
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

def format_mediapipe(seq):
    '''
    Transformation Function. Formats the normalized keypoints as per mediapipe output.
    Gets rid of keypoint 1 (Palm) which is not provided by mediapipe.
    Calculates the relative hand vectors and appends them to the absolute coordinates.
    '''
    new_seq = torch.zeros((len(seq), 74))  # 42 absolute + 32 relative
    for i, iseq in enumerate(seq):
        # Absolute
        new_seq[i] = torch.cat([iseq[0:2], iseq[4:], torch.zeros(32)])

        # Relative
        for j in range(4):
            # calculate L01, L12, L23, L34
            new_seq[i][42+2*j] = new_seq[i][2*j+2] - new_seq[i][2*j] #L__X
            new_seq[i][42+2*j+1] = new_seq[i][2*j+3] - new_seq[i][2*j+1] #L__Y

        for j in range(3):
            # calculate L56, L67, L78
            new_seq[i][50+2*j] = new_seq[i][2*j+12] - new_seq[i][2*j+10]
            new_seq[i][50+2*j+1] = new_seq[i][2*j+13] - new_seq[i][2*j+11]

            # calculate L910, L1011, L1112
            new_seq[i][56+2*j] = new_seq[i][2*j+20] - new_seq[i][2*j+18]
            new_seq[i][56+2*j+1] = new_seq[i][2*j+21] - new_seq[i][2*j+19]

            # calculate L1314, L1415, L1516
            new_seq[i][62+2*j] = new_seq[i][2*j+28] - new_seq[i][2*j+26]
            new_seq[i][62+2*j+1] = new_seq[i][2*j+29] - new_seq[i][2*j+27]

            # calculate L1718, L1819, L1920
            new_seq[i][68+2*j] = new_seq[i][2*j+36] - new_seq[i][2*j+34]
            new_seq[i][68+2*j+1] = new_seq[i][2*j+37] - new_seq[i][2*j+35]

    return new_seq


def read_data(seed_val):
    ''' Reads data from SHREC2017 dataset files. '''
    # Change this as per your system
    base_directory = "/home/sriramsk/Desktop/HandGestureDataset_SHREC2017"

    gesture_dir = ['gesture_'+str(i) for i in range(1, 15)]
    gesture_arr = []
    target_arr = []
    gesture_no = 1

    for gesture in [os.path.join(base_directory, i) for i in gesture_dir]: # for each gesture
        for finger in ['/finger_1', '/finger_2']:
            for subject in os.listdir(gesture+finger):
                for essai in os.listdir(gesture+finger+'/'+subject):
                    data = np.loadtxt(gesture+finger+'/'+subject+'/'+essai+'/skeletons_image.txt')
                    gesture_arr.append(data)
                    target_arr.append(gesture_no)
        gesture_no += 1
    return train_test_split(gesture_arr, target_arr,
                            test_size=0.2, random_state=seed_val)

def main():
    ''' Main '''

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    EPOCHS = 10
    SEED_VAL = 42
    init_seed(SEED_VAL)
    writer = SummaryWriter('logs')

    ##################
    # INPUT PIPELINE #
    ##################

    train_x, test_x, train_y, test_y = read_data(SEED_VAL)
    # Custom transforms to prepare data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Lambda(format_mediapipe),
    ])
    train_loader = DataLoader(ShrecDataset(train_x, train_y, transform))
    test_loader = DataLoader(ShrecDataset(test_x, test_y, transform))

    ############
    # TRAINING #
    ############

    for batch_no, batch in enumerate(train_loader):
        input_data = batch[0].to(device)
        target = batch[1].to(device)


if __name__ == '__main__':
    main()
