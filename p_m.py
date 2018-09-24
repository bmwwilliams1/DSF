import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy
import matplotlib.mlab as mlab
from scipy.stats import lognorm
from scipy.stats import expon

# This file contains all the helper functions for graphing and producing statistics
# of the image properties


#  This function plots a single image as a heatmap of a matrix.
#  It is called within loops for graphing multiple neurons.
def plot(size, ax,a,b,w,min=0,max=1):
    image = np.reshape(w,(size,size))
    im = ax.pcolormesh(a,-b, image,vmin=min,vmax=max)
    # if i==0:
    #     fig.colorbar(im)
    ax.axis('tight')

# This function produces one of the sparsity metrics,
# counting the number of elements in the matrix that are < 5% of the maximum
def sparse_met(w1,w2):

    w1_count = len(w1)*len(w1[0])
    w2_count = len(w2)*len(w2[0])
    epsilon = 0.001

    w1_zero = np.zeros((1,len(w1[0])),dtype=float)
    w2_zero = np.zeros((1,len(w2[0])),dtype=float)


    for i in range(0,len(w1[0])):
        for j in range(0,len(w1)):
            if (w1[j][i] > epsilon and w1[j][i]>(w1[:,i].max()*0.05)):
                w1_zero[0,i] = w1_zero[0,i] + 1
            if (w2[j][i] > epsilon and w2[j][i]>(w2[:,i].max()*0.05)):
                w2_zero[0,i] = w2_zero[0,i] + 1

    w1_zero = w1_zero*100/ (len(w1))
    w2_zero = w2_zero*100 / (len(w2))

    np.set_printoptions(precision=4)


# This function floors a matrix by setting any negative values to 0
def floor(w):

    w1 = np.zeros((len(w),len(w[0])),dtype=float)
    for i in range(0,len(w1)):
        for j in range(0,len(w1[0])):
            if w[i][j]>0:
                w1[i][j]=w[i][j]
            else:
                w1[i][j]=0
    return w1

# This function rescales values to [0,1]
def rescale(w):
    w = w - (w.min())
    w = w / (w.max())
    return w

# This function calculates the Shannon entropy of a raw matrix.
# It is used in combination with bin_total_entropy to bin and calculate entropy
def entropy(w):
    entropy = 0
    uniques = np.unique(w)
    for elem in uniques:
        count = sum([1 if i==elem else 0 for i in w])
        p = count/len(w)

        if p > 0:
            entropy += (-p * math.log(p, 2))

    return entropy

# This calculates bins the values of a matrix into 100 bins, and calculates the resulting
# Shannon entropy
def bin_total_entropy(w):

    bins = 100
    total = np.zeros((1,len(w[0])),dtype=float)

    for i in range(0,len(w[0])):
        rescaled = rescale(w[:,i])
        rescaled_bin = bin(rescaled, 100)
        total[0,i] = entropy(rescaled_bin)

    return total

# This function bins an array values into the chosen number of bins
def bin(vector, bins):
    bint = np.linspace(0,1,bins+1)
    new_vec = np.zeros((len(vector),1),dtype=float)

    for elem in range(0,len(vector)):
        for box in range(0,len(bint)-1):
            if (vector[elem]>=bint[box] and vector[elem]<bint[box+1]):
                new_vec[elem]=bint[box]
                continue

    return new_vec


# This function calculates the elementwise Euclidean distance between
# two vectors.
def distance(a,b):

    if not (len(a)==len(b)):
        print('Error: vectors unequal length')
        return 0
    dist = 0
    for elem in range(0,len(a)-1):
        dist = dist+(a[elem]-b[elem])*(a[elem]-b[elem])
    return np.sqrt(dist)

# This function produces an columnwise average of a matrix
def mean(weights):

    mean = np.zeros((len(weights[:,0]),),dtype=float)
    i= 0
    for pixel in weights:
        mean[i] = sum(pixel)/len(pixel)
        i=i+1

    return mean

# This function produces the mean and covariances of a matrix.
def metrics(weights):

    mean = np.zeros((len(weights[:,0]),),dtype=float)
    i= 0
    for pixel in weights:
        mean[i] = sum(pixel)/len(pixel)
        i=i+1


    diff = np.zeros((len(weights),len(weights[0])),dtype=float)

    for row in range(0,len(weights)):
        for col in range(0,len(weights[0])):
            diff[row,col]=weights[row,col]-mean[row]


    cov = np.zeros((len(weights[0]),len(weights[0])),dtype=float)

    for row in range(0,len(weights[0])):
        for col in range(0,len(weights[0])):
            for ex in range(0,len(weights)):
                cov[row,col] = cov[row,col]+(diff[ex,row]*diff[ex,col])

    cov = cov/(len(weights)-1)

    return mean, cov

# This graphing function produces the image weights (DSF, ANN, NN-ANN) all side by side for use in the report.
def graph_report(size, offset, w1, w2 = None):

    if(w2 is None):
        print('Error: No w2 matrix!')
        return

    if (w1.shape!=w2.shape):
        print('Error: Matrices are not the same size!')
        return

    fig,ax = plt.subplots(8,3)#len(w1[0]),3)

    row = 0

    a = np.linspace(1, size,size)
    b = np.linspace(1, size,size)
    a, b = np.meshgrid(a,b)
    i = 0
    for sub in ax:
        # for subi in sub:
        p_m.plot(size,sub[0],b,a,w1[:,i+offset])
        if i==0:
            sub[0].set_title("DSF")
        p_m.plot(size,sub[1],b,a,w2[:,i+offset],min=-1)
        if i==0:
            sub[1].set_title("ANN")
        p_m.plot(size,sub[2],b,a,w2[:,i+offset])
        if i==0:
            sub[2].set_title("NN-ANN")
        sub[1].get_xaxis().set_visible(False)
        sub[1].get_yaxis().set_visible(False)
        sub[0].get_xaxis().set_visible(False)
        sub[0].get_yaxis().set_visible(False)
        sub[2].get_xaxis().set_visible(False)
        sub[2].get_yaxis().set_visible(False)
        i=i+1
    plt.show()

# This function produces (and optionally saves) the mean and covariance matrix of a single weight image.
def run_metrics(save_covariance, run_metrics, w1):

    mean, cov = p_m.metrics(w1)

    if (save_covariance==True):
        np.savetxt("cov.csv", cov, delimiter=",")
        # plot the covariance image to the user for confirmation
        fig,ax = plt.subplots(1,1)
        n_points = 128
        a = np.linspace(1, 128, n_points)
        b = np.linspace(1, 128, n_points)
        a, b = np.meshgrid(a, b)
        im = ax.pcolormesh(a, -b, cov)
        ax.axis('tight')
        plt.show()
    # ================================================


#This is a more flexible graphing function for quickly visualing a set of neuron weight images.
def graph_components(size, offset, w1, w2 = None):

    #  Graphing the components, here we can graph a variable number of the graphs individually

    n_points = size
    a = np.linspace(1, n_points, n_points)
    b = np.linspace(1, n_points, n_points)
    a, b = np.meshgrid(a,b)

    if (len(w1[0])==16 or len(w1[0]) == 64):
        dim = int(np.sqrt(len(w1[0])))
        fig,ax = plt.subplots(dim,dim)

    else:
        fig,ax = plt.subplots(8,4)

    i = 0
    for sub in ax:
        for subi in sub:
            p_m.plot(size,subi,b,a,w1[:,i+offset])
            subi.get_xaxis().set_visible(False)
            subi.get_yaxis().set_visible(False)
            i=i+1
    plt.show()
