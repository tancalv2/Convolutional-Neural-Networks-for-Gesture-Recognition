import numpy as np
import scipy.signal as sp
import torch
import matplotlib.pyplot as plt

from time import time
from torch.utils.data import DataLoader
from csv2numpy import convert_to_numpy
from normalize_data import normalize_data
from data_split import data_split
from dataset import GestureDataset
from model import CNN


'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

# =================================== LOAD DATA AND MODEL =========================================== #


def load_data(batch_size):
    ######

    # 3.1 YOUR CODE HERE
    train_dataset = GestureDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = GestureDataset(val_data, val_label)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ######

    return train_loader, val_loader


def load_model(lr):
    ######

    # 3.2 YOUR CODE HERE

    loss_fnc = torch.nn.CrossEntropyLoss()
    try:
        model = CNN(train_data.shape[1]).cuda()
    except:
        model = CNN(train_data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ######

    return model, loss_fnc, optimizer


# =================================== EVALUATION MODEL =========================================== #


def evaluate(model, val_loader):
    ######

    # 3.4 YOUR CODE HERE

    total_corr = 0
    for data_v in val_loader:
        gestures, labels = data_v
        prediction = model(gestures)
        for e in range(len(prediction)):
            guess = torch.argmax(prediction[e])
            try:
                answer = labels[e].cuda()
            except:
                answer = labels[e]
            if guess == answer:
                total_corr += 1

    ######

    return float(total_corr)/len(val_loader.dataset)


def test():

    # load model
    model = torch.load('./data/model.pt')
    # model = torch.load('./data/model.pt', map_location='cpu')

    # test if model has correct validation accuracy
    # train_loader, val_loader = load_data(bs)
    # print(evaluate(model, val_loader))

    # load data set
    test_data = np.load('./data/test_data.npy')

    # normalize the data
    for e in range(test_data.shape[0]):
        for index in range(5):
            # average over time as well, sum over all 100, then subtract mean divide by std
            test_data[e, :, index] = (test_data[e, :, index] - test_data[e, :, index].mean()) \
                                     / test_data[e, :, index].std()

    # swap the input channel and sensor value columns since the conv1d requires input channels to be in second column
    test_data = test_data.swapaxes(1, 2)

    # create predictions array
    predictions = [0] * test_data.shape[0]

    # evaluate model on test set
    test_dataset = GestureDataset(test_data, predictions)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    counter = 0
    for data_t in test_loader:
        gestures, labels = data_t
        prediction = model(gestures)
        for e in range(len(prediction)):
            predictions[counter] = (torch.argmax(prediction[e]))
            counter += 1

    # save as txt file
    np.savetxt('./data/predictions.txt', predictions)


# =================================== VISUALIZE TRAINING AND VALIDATION =========================================== #


def plot_data(valid_acc, train_acc, epoch, max_loc):
    ######

    # 3.5 YOUR CODE HERE

    plt.figure()
    plt.title("Accuracy vs. Epochs (BS={}, lr={}, Max_Epoch={})".format(bs, lr, epoch), fontsize=12)
    plt.plot(valid_acc, label="Validation")
    plt.plot(train_acc, label="Training")
    plt.plot(max_loc, [max(valid_acc)], marker='o', color="red", label="Max Acc: {}"
             .format(int(max(valid_acc)*10000)/100))
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc=4, fontsize=10)  # lower right
    plt.savefig("./plots/acc_vs_epoch_bs_{}_lr_{}_max_epoch_{}.png".format(bs, lr, epoch))
    plt.show()

    #######


def plot_data_steps(valid_acc, train_acc, max_loc):
    ######

    plt.figure()
    plt.title("Accuracy vs. Steps (BS={}, lr={}, Step Size = {})".format(bs, lr, steps), fontsize=12)
    plt.plot(valid_acc, label="Validation")
    plt.plot(train_acc, label="Training")
    plt.plot(max_loc, [max(valid_acc)], marker='o', color="red", label="Max Acc: {}"
             .format(int(max(valid_acc)*10000)/100))
    plt.xlabel("Number of Steps", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc=4, fontsize=10)  # lower right
    plt.savefig("./plots/acc_vs_step_bs_{}_lr_{}_steps_{}_loc_{}.png".format(bs, lr, steps, max_loc))
    plt.show()

    ######

# ============================================== TRAINING LOOP ====================================================== #


def train():
    ######

    # 3.3 YOUR CODE HERE

    train_loader, val_loader = load_data(bs)
    model, loss_fnc, optimizer = load_model(lr)
    MaxEpoch = epochs
    start_time = time()

    # variables for every epoch
    valid_acc = [0]
    max_valid_acc = 0
    max_loc = 0
    train_acc = [0]
    train_acc_is_one = 0

    # variables for every step size
    t = 0
    counter = 0
    valid_acc_steps = [0]
    max_valid_acc_steps = 0
    max_loc_steps = 0
    train_acc_steps = [0]
    total_corr_batch = 0

    for epoch in range(MaxEpoch):
        accum_loss = 0
        total_corr = 0
        for train_data in train_loader:
            gestures, labels = train_data
            optimizer.zero_grad()
            prediction = model(gestures)
            try:
                batch_loss = loss_fnc(input=prediction, target=labels.long().cuda())
            except:
                batch_loss = loss_fnc(input=prediction, target=labels.long())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            for e in range(len(prediction)):
                guess = torch.argmax(prediction[e])
                try:
                    answer = labels[e].cuda()
                except:
                    answer = labels[e]
                if guess == answer:
                    total_corr += 1
                    total_corr_batch += 1

            # evaluate per step size
            if(t + 1) % steps == 0:
                valid_acc_steps.append(evaluate(model, val_loader))
                train_acc_steps.append(total_corr_batch/steps/bs)
                counter += 1
                total_corr_batch = 0

                # keep track of the highest accuracy done by the model
                # save and plot only if model achieves a certain threshold of accuracy (89% since that is my max)
                if valid_acc_steps[counter] > max_valid_acc_steps:
                    max_valid_acc_steps = valid_acc_steps[counter]
                    max_loc_steps = counter
                    if max_valid_acc_steps > max_valid_acc and max_valid_acc_steps > 0.89:
                        # save model as it can potentially be highest accuracy
                        # note: models were saved in this manor such that it can be easily identified which model it is
                        torch.save(model, './models/model_BS_{}_lr_{}_loc_{}.pt'.format(bs, lr, max_loc))
                        # plot whenever a model is saved
                        plot_data(valid_acc, train_acc, epoch + 1, max_loc)
                        # smoothen step values due to more fluctuation
                        valid_acc_steps_plot = sp.savgol_filter(valid_acc_steps, 3, 1)
                        train_acc_steps_plot = sp.savgol_filter(train_acc_steps, 3, 1)
                        plot_data_steps(valid_acc_steps_plot, train_acc_steps_plot, max_loc_steps)
            t = t + 1

        # evaluate per epoch
        valid_acc.append(evaluate(model, val_loader))
        train_acc.append(float(total_corr) / len(train_loader.dataset))

        # keep track of the highest accuracy done by the model
        # save and plot only if model achieves a certain threshold of accuracy (89% since that is my max)
        if valid_acc[epoch+1] > max_valid_acc:
            max_loc = epoch + 1
            max_valid_acc = valid_acc[epoch+1]
            print("new max: {}".format(max_valid_acc))
            if max_valid_acc > 0.89:
                # save model as it can potentially be highest accuracy
                # note: models were saved in this manor such that it can be easily identified which model it is
                torch.save(model, './models/model_BS_{}_lr_{}_epoch_{}.pt'.format(bs, lr, epoch+1))
                # plot whenever a model is saved
                plot_data(valid_acc, train_acc, epoch+1, max_loc)
                # smoothen step values due to more fluctuation
                valid_acc_steps_plot = sp.savgol_filter(valid_acc_steps, 3, 1)
                train_acc_steps_plot = sp.savgol_filter(train_acc_steps, 3, 1)
                plot_data_steps(valid_acc_steps_plot, train_acc_steps_plot, max_loc_steps)

        # print records
        print("Epoch: {}| Loss: {} | Train acc: {} |  Valid acc: {} | Valid max: {}"
              .format(epoch + 1, accum_loss / 100, train_acc[epoch+1], valid_acc[epoch+1], max_valid_acc))

        # check if training accuracy has reached peak, if so, only loss function is being optimized, over-fitting will
        # occur from this point out, allow for a maximum of 20 increments, then exiting training (I choose it to be 20)
        if train_acc[epoch] > 0.99999999:
            train_acc_is_one += 1
            if train_acc_is_one == 20:
                break
        else:
            train_acc_is_one = 0

    print("\n")
    end_time = time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    print("Final Train acc: {}".format(float(total_corr) / len(train_loader.dataset)))
    print("Final Validation acc: {}".format(evaluate(model, val_loader)))

    # plot final plot of both epoch and step size
    plot_data(valid_acc, train_acc, epoch+1, max_loc)
    # smoothen step values due to more fluctuation
    valid_acc_steps = sp.savgol_filter(valid_acc_steps, 3, 1)
    train_acc_steps = sp.savgol_filter(train_acc_steps, 3, 1)
    plot_data_steps(valid_acc_steps, train_acc_steps, max_loc_steps)

    ######


if __name__ == "__main__":
    # seed of 6 was found to be most optimal
    seed = 6
    split_size = 0.2
    # convert all files, normalize and split data.
    # convert_to_numpy()        # commented out since files do not need be constantly converted
    # normalize_data()          # commented out since files do not need be constantly normalized
    data_split(seed, split_size)
    # load data
    train_data = np.load('./data/train_data.npy')
    train_label = np.load('./data/train_label.npy')
    val_data = np.load('./data/val_data.npy')
    val_label = np.load('./data/val_label.npy')

    # set hyperparameters
    lr = 0.005175
    steps = 100
    epochs = 500
    bs = 32

    # train if necessary
    train()

    # test if necessary
    # test()




