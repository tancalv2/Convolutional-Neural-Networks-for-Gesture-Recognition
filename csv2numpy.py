import numpy as np
import pandas as pd

'''
    Save the data in the .csv file, save as a .npy file in ./data


'''

# =================================== CONVERT DATASET =========================================== #

######

# 2.1 YOUR CODE HERE


def convert_to_numpy():
    labels = []
    instances = [None]*5590
    i = 0

    # 43 students
    for student in range(43):
        # 26 letters
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            # 5 entries per letter
            for index in [1, 2, 3, 4, 5]:
                data = pd.read_csv("./unnamed_train_data/student{}/{}_{}.csv".format(student,letter,index), header=None)
                data = data.drop(columns=0)

                # drop z-accel from dataset
                # data = data.drop(columns=2)

                labels.append('{}'.format(letter))
                instances[i] = np.array(data)
                # print('student {}: {}_{}'.format(student, letter, index))
                i += 1

    labels = np.array(labels)

    np.save('./data/instances.npy', instances)
    np.save('./data/labels.npy', labels)
    print("Files converted")

######
