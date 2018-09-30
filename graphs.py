
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import p_m
from scipy.stats import ks_2samp


# ============ MAIN ===============

def main():

    # neurons = 16
    im_size = 28

    neurons = [16,32,64]

    for i in range(0,len(neurons)):
        w1 = np.genfromtxt('./MNIST_sigmoid_models/DSF/layer1/weights_' + str(neurons[i]) + '_1.csv',delimiter=",")
        w2 = np.genfromtxt('./MNIST_sigmoid_models/ANN/layer1/weights_' + str(neurons[i]) + '_1.csv',delimiter=",")

        offset = 0
        while(offset < neurons[i]):
            print('offset: '+str(offset))
            p_m.graph_report(neurons =neurons[i],size=im_size, offset=offset, w1=w1, w2 = w2)
            offset +=8
    # p_m.graph_components(size=im_size, offset=0, w1=w1, w2 = None)
    # p_m.graph_report(size=im_size, offset=16, w1=w1, w2 = w2)
    # mean, cov = p_m.metrics(w1)


if __name__ == "__main__":
    main()
