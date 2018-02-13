import scipy
import scipy.io
import sys
import numpy as np
import math

def read_mat_file():
	# Assumes the hw1data.mat file is in the same folder as the code - if not, change here. 
    mat_file = sys.path[0] + "/hw1data.mat"
    mat = scipy.io.loadmat(mat_file)

    return mat