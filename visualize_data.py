import numpy as np
import matplotlib.pyplot as plt

'''
    Visualize some samples.
'''

# =================================== VISUALIZE DATA =========================================== #

######

# 2.2 YOUR CODE HERE


def visualize_data_func(ax, ay, az, wx, wy, wz, letter, student):
    plt.figure()
    plt.title('Student {}, Gesture {}'.format(student+1, letter), fontsize=22)
    plt.plot(ax, label="x-acceleration")
    plt.plot(ay, label="y-acceleration")
    plt.plot(az, label="z-acceleration")
    plt.plot(wx, label="pitch")
    plt.plot(wy, label="roll")
    plt.plot(wz, label="yaw")
    plt.xlabel("Time Interval", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend(loc='best', fontsize=10)
    plt.savefig("./plots/student_{}_gesture_{}.png".format(student+1, letter))
    plt.show()


def main():
    instances = np.load('./data/instances.npy')
    student = 0
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    for n in range(2):          # every letter is 5 away from each other
        for i in range(3):      # every student is 130 away from each other
            ax = instances[130*i + 5*n, 0:100, 0]
            ay = instances[130*i + 5*n, 0:100, 1]
            az = instances[130*i + 5*n, 0:100, 2]
            wx = instances[130*i + 5*n, 0:100, 3]
            wy = instances[130*i + 5*n, 0:100, 4]
            wz = instances[130*i + 5*n, 0:100, 5]
            visualize_data_func(ax, ay, az, wx, wy, wz, letter[n], student)
            student = (student + 1) % 3


if __name__ == "__main__":
    main()

    ######
