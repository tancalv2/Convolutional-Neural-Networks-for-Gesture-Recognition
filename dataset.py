import torch.utils.data as data

'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

# =================================== NORMALIZE DATASET =========================================== #

######

# 3.1 YOUR CODE HERE


class GestureDataset(data.Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        labels = self.labels[index]
        return features, labels

######

