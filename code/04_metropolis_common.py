import numba
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)
random.seed(1234) # python random RNG (used in jit-functions)
np.random.seed(1234) # numpy RNG
