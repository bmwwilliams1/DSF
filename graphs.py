
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import p_m
from scipy.stats import ks_2samp


# ============ MAIN ===============

def main():

    neurons = 16
    im_size = 32

    w1 = np.genfromtxt('./dsf_wt/weights_mathdsf512_1.csv',delimiter=",")
    w2 = np.genfromtxt('./ffn_wt/weights_512_1.csv',delimiter=",")

    graph_report(size=im_size, offset=0, w1=w1, w2 = w2)



if __name__ == "__main__":
    main()
