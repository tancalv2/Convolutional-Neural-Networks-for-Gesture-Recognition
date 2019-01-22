import numpy as np
import matplotlib.pyplot as plt

'''
    Visualize some basic statistics of our dataset.
'''

# =================================== VISUALIZE DATA =========================================== #

######

# 2.3 YOUR CODE HERE


def bin_data_func(mean, std, letter):
    w = 0.80
    ind = np.array(['a_x', 'a_y', 'a_z', 'w_x', 'w_y', 'w_z'])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(ind, mean, w)
    ax.errorbar(ind, mean, yerr=std, fmt='o')
    ax.set_xticklabels(ind, fontsize=16)
    plt.title('Gesture {}'.format(letter), fontsize=22)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Average Sensor Value", fontsize=18)
    plt.savefig("./data/bar_gesture_{}.png".format(letter))
    plt.show()


def main():
    instances = np.load('./data/instances.npy')

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    a = [None]*(43*5)
    mean = [None] * 3     # assignment asks for 3, however 26 is also easily computable
    std = [None] * 3

    for C in range(3):          # number of gestures
        x = 0
        for i in range(43):  # every student is 130 away from each other
            for j in range(5):  # 5 same gestures per person
                a[x] = np.array(instances[130 * i + j + 5 * C, 0:100])
                x += 1
        a = np.array(a)
        mean_gest = [None] * 6
        std_gest = [None] * 6
        for col in range(6):
            mean_gest[col] = a[:, :, col].mean()
            std_gest[col] = a[:, :, col].std()
        mean[C] = mean_gest
        std[C] = std_gest
        bin_data_func(mean[C], std[C], letter[C])


if __name__ == "__main__":
    main()

    ######
