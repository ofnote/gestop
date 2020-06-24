'''
Describes the implementation of the training procedure for gesture net
'''
import time
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import GestureNet, GestureDataset


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

    loaded_data = DataLoader(GestureDataset(formatted_data, formatted_target), batch_size=batchsize)
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    EPOCHS = 10
    SEED_VAL = 42
    init_seed(SEED_VAL)
    writer = SummaryWriter('logs')

    ##################
    # INPUT PIPELINE #
    ##################

    # Read and format the csv
    df = pd.read_csv("data/static_gestures_data.csv")
    train, test = train_test_split(df, test_size=0.1, random_state=SEED_VAL)
    train_X, train_Y = split_dataframe(train)
    test_X, test_Y = split_dataframe(test)

    # One Hot Encoding of the target classes
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)

    le = LabelEncoder()
    le.fit(train_Y)

    # Store encoding to disk
    le_name_mapping = dict(zip([int(i) for i in le.transform(le.classes_)], le.classes_))
    print(le_name_mapping)
    with open('data/gesture_mapping.json', 'w') as f:
        f.write(json.dumps(le_name_mapping))


    train_Y = le.transform(train_Y)
    test_Y = le.transform(test_Y)

    BATCH_SIZE = 64
    train_loader = format_and_load(train_X, train_Y, BATCH_SIZE)
    test_loader = format_and_load(test_X, test_Y, BATCH_SIZE)


    ############
    # TRAINING #
    ############

    OUTPUT_CLASSES = 6
    INPUT_DIM = 49 #refer make_vector() to verify input dimensions

    gesture_net = GestureNet(INPUT_DIM, OUTPUT_CLASSES)
    optimizer = torch.optim.Adam(gesture_net.parameters(), lr=5e-3)
    criterion = torch.nn.CrossEntropyLoss()
    gesture_net.cuda()

    training_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    for epoch in range(EPOCHS):

        gesture_net.train()

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
        start_time = time.time()
        TOTAL_LOSS = 0

        for batch_no, batch in enumerate(train_loader):
            input_data = batch[0].to(device)
            target = batch[1].to(device)

            gesture_net.zero_grad()
            output = gesture_net(input_data.float())

            loss = criterion(output, target.long())
            TOTAL_LOSS += loss.item()

            loss.backward()
            optimizer.step()

        #Logging the loss and accuracy in Tensorboard
        avg_train_loss = TOTAL_LOSS / len(train_loader)
        training_loss_values.append(avg_train_loss)

        for name, weights in gesture_net.named_parameters():
            writer.add_histogram(name, weights, epoch)

        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        print("Average training loss: {0:.2f}".format(avg_train_loss))

        # Validation

        gesture_net.eval()

        test_loss, test_accuracy = 0, 0
        NB_EVAL_STEPS = 0

        for batch_no, batch in enumerate(test_loader):

            input_data = batch[0].to(device)
            target = batch[1].to(device)

            with torch.no_grad():
                output = gesture_net(input_data.float())
                loss = criterion(output, target.long())

            target = target.to('cpu').numpy()
            output = output.to('cpu').numpy()

            tmp_eval_accuracy = calc_accuracy(target, output)
            test_accuracy += tmp_eval_accuracy
            test_loss += loss.item()

            NB_EVAL_STEPS += 1

        avg_valid_acc = test_accuracy/NB_EVAL_STEPS
        avg_valid_loss = test_loss/NB_EVAL_STEPS
        validation_loss_values.append(avg_valid_loss)
        validation_accuracy_values.append(avg_valid_acc)

        writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
        writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
        writer.flush()

        print("Avg Val Accuracy: {0:.2f}".format(avg_valid_acc))
        print("Average Val Loss: {0:.2f}".format(avg_valid_loss))
        print("Time taken by epoch: {0:.2f}".format(time.time() - start_time))

    ################
    # SAVING MODEL #
    ################

    PATH = 'models/gesture_net'
    torch.save(gesture_net.state_dict(), PATH)


if __name__ == '__main__':
    main()
