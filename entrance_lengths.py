import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit, newton, minimize
matplotlib.use('TkAgg')

# Data Import
data = np.load('data/data.npy')

# Constants
rho = 998.2
mu = 0.0010016
inh202pa = 248.84
d = 0.02121
l = 0.844
mtoin = 39.3701

# Easy Variables
dP = data[:, 1]
vel = data[:, 0]

# Reynolds Calculation
Res = rho*vel*d/mu

# Friction Factor Calculation
Cf = data[:, 1]/(0.5*rho*(l/d)*data[:, 0]**2)

# Calculate Entrance Length Conditions
ent = 1.359*d*Res**0.25

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(6.5, 4), sharey='row', layout='constrained')


# Titles
fig.suptitle('Entrance Length vs. Reynolds Number', fontsize=11)

# Styles
from matplotlib import font_manager

font_path = 'EBGaramond-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


plt.plot(Res, ent*mtoin,
                       linewidth=1,
                       color='black',
                       label='Experimental')
plt.axhline(y=10*d*mtoin, linewidth=1.0, color='k', linestyle='dotted')
plt.axhline(y=40*d*mtoin, linewidth=1.0, color='k', linestyle='dotted')
plt.axhline(y=27, linewidth=1.0, color='k', linestyle='dotted')
plt.text(0.6*max(Res), 1.05*10*d*mtoin, 'General Criterion '+r'$L_{ent} = 10D_H$')
plt.text(0.6*max(Res), 0.96*40*d*mtoin, 'Nikuradse Criterion '+r'$L_{ent} = 40D_H$')
plt.text(0.6*max(Res), 0.96*27, 'Actual Entrance Length: 27')

plt.xlabel('Re')
plt.ylabel('Entrance Length (in)')
plt.show()