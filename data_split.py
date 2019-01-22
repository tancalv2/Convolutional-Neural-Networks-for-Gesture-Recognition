import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

'''
    This file contains the splitting of the training and validation set
'''
# =================================== TRAINING AND VALIDATION DATA SPLIT =========================================== #

# 2.5 YOUR CODE HERE


def data_split(seed, split_size):
    # LOAD DATA
    data = np.load('./data/normalized_data.npy')
    # swap the input channel and sensor value columns since the conv1d requires input channels to be in second column
    data = data.swapaxes(1, 2)
    # LOAD LABELS
    label = np.load('./data/labels.npy')
    # ENCODE LABELS
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    # SPLIT DATA AND LABELS INTO TRAINING AND VALIDATION SET
    train_data,val_data,train_label,val_label = train_test_split(data, label, test_size=split_size, random_state=seed)

    # SAVE
    np.save('./data/train_data.npy', train_data)
    np.save('./data/val_data.npy', val_data)
    np.save('./data/train_label.npy', train_label)
    np.save('./data/val_label.npy', val_label)
    print("Data has now been split")

######

