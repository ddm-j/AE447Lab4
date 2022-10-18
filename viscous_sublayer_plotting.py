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

# Wall Shear
tau = dP * d/(4*l)#2*Cf*rho*vel**2

print(tau)

# Boundary Layer Defintions
u, ut, k, nu, y, y0 = sym.symbols("u, u_t, k, nu, y, y_0")
eq = (ut/k)*sym.log(y/y0) - u
eq = eq.subs(y0, nu/(9*ut)).simplify()
ut_sol = sym.solve(eq, ut)[0]

# Quadratic Bezier
yp1, yp2, C, t, U_inf, r = sym.symbols('yp1, yp2, C, t, U_inf, r')
p0 = sym.Matrix([yp1, yp1])
p2 = sym.Matrix([yp2, (1/k)*sym.log(yp2) + C])
B = (1/k)*(sym.log(yp2) - 1) + C
A = B/(1 - 1/(k*yp2))
p1 = sym.Matrix([A, A])

# Bezier Function
Bz = (1-t)*((1-t)*p0 + t*p1) + t*((1-t)*p1 + t*p2)

# Viscous Sublayer Function
u_v = y*ut**2/nu

# Log - Law Layer Function
u_l = (1/k)*sym.log(y)+5.0

ut_data = []
for i in range(len(Res)):
    # Calculate the Required Wall Shear
    ut_calc = float(ut_sol.subs(
        {
            k: 0.41,
            u: vel[i],
            y: d/2,
            nu: mu/rho
        }
    ))

    ut_data.append(ut_calc)

tau_loglaw = rho*np.array(ut_data)**2

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(6.5, 4), sharey='row', layout='constrained')

# Titles
fig.suptitle('Comparison of Experimental and Log Wall Shear Stress', fontsize=11)

# Styles
from matplotlib import font_manager

font_path = 'EBGaramond-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


# Plot Experimental Wall Shear
#label=r'$U_\infty = {0}$'.format(key),
axs.plot(Res, tau,
                   linewidth=1,
                   color='black', label='Experimental Wall Shear')
axs.plot(Res, tau_loglaw,
                   linewidth=0.75,
                   color='black', label='Log-Law Wall Shear', linestyle='dashed')

# Formatting
axs.set_ylabel('Wall Shear Stress (Pa)')
axs.set_xlabel('Re')

# Set X-Axis Scale
#axs.set_xscale('log')

handles, labels = axs.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=plt.gcf().transFigure)
plt.show()
