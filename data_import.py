import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Constants
rho = 998.2
mu = 0.0010016
inh202pa = 248.84
d = 0.02121

# Import the Raw Data and Process it
data = np.loadtxt('data/raw_data.txt')
vel = data[0, :]
P = inh202pa*np.mean(data[1:, :], axis=0)
Re = rho*vel*d/mu

# Export the Data
data_exp = np.zeros((len(Re), 2))
data_exp[:, 0] = vel
data_exp[:, 1] = P

np.save('data/data.npy', data_exp)