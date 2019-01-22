import numpy as np

'''
    Normalize the data, save as ./data/normalized_data.npy
'''

# =================================== NORMALIZE DATASET =========================================== #

######

# 2.4 YOUR CODE HERE


def normalize_data():
    instances = np.load('./data/instances.npy')
    for e in range(43*26*5):
        for index in range(5):
            # average over time as well, sum over all 100, then subtract mean divide by std
            instances[e, :, index] = (instances[e, :, index]-instances[e, :, index].mean())/instances[e, :, index].std()
    np.save('./data/normalized_data.npy', instances)
    print("Data normalized")

######
