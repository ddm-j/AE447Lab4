import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit, newton, minimize
matplotlib.use('TkAgg')

# Data Import
data = np.load('data/data.npy')

# Constants
rho = 998.2
mu = 0.0010016
nu = mu/rho
print(nu)
inh202pa = 248.84
d = 0.02121
l = 0.844

# Easy Variables
dP = data[:, 1]
vel = data[:, 0]

# Velocity Curve fit
def vel_para(x, c):
    return c*x**2

popt, pcov = curve_fit(vel_para, vel, dP)
print(popt)
dP_fit = popt[0]*vel**2

# Reynolds Calculation
Res = rho*vel*d/mu

# Friction Factor Calculation
Cf = data[:, 1]/(0.5*rho*(l/d)*data[:, 0]**2)

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(6.5, 4), sharey='row', layout='constrained')

# Titles
fig.suptitle('Pressure Drop vs Velocity', fontsize=11)

# Styles
from matplotlib import font_manager

font_path = 'EBGaramond-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


# Plot Experimental Data
axs.plot(vel, dP,
               linewidth=1,
               color='black',
               label='Experimental')

axs.set_xlabel(r'$U_\infty (\frac{m} {s})$', fontsize=10)
axs.set_ylabel(r'$\Delta P (Pa)$', fontsize=10)
handles, labels = axs.get_legend_handles_labels()

#fig.legend(handles, labels, loc='lower right')
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), bbox_transform=plt.gcf().transFigure)

plt.savefig('charts/pressureDrop.png', dpi=400, bbox_inches='tight')
plt.show()