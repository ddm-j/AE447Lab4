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

# Easy Variables
dP = data[:, 1]
vel = data[:, 0]

# Reynolds Calculation
Res = rho*vel*d/mu

# Friction Factor Calculation
Cf = data[:, 1]/(0.5*rho*(l/d)*data[:, 0]**2)

# Min/Max in Dict
def dict_limit(d, func):
    res = []
    for key in d.keys():
        res.append(func(d[key]))

    return func(np.array(res))

def roughness_function(Cfs, eps, dh=d):
    # B Parameter
    B = 1/(1.93*np.sqrt(Cfs)) + np.log10(1.9*eps/(np.sqrt(8)*dh))
    return B


def reynolds_roughness(Re, Cfs, eps, dh=d):
    # Reynolds Roughness
    Rs = (eps/(np.sqrt(8)*dh))*(Re*np.sqrt(Cfs))
    return Rs


# Surface Roughness
# Variable Definitions
#f, eta, D_h, Re_f = sym.symbols('f eta D_h Re_f')
#eq = -2*sym.log10(eta/(3.7*D_h) + 2.51/(Re_f*sym.sqrt(f))) - 1/sym.sqrt(f)

# Fit The Data


def friction_model(f, Re, eta, model='cole', dh=d):
    # Colebrook-white
    if model == 'cole':
        fun = -2 * np.log10(eta / (3.7 * dh) + 2.51 / (Re * f ** 0.5)) - 1 / f ** 0.5

    # Nikuradse - Rough
    elif model == 'nikr':
        fun = 1.14 - 2 * np.log10(eta / dh) - 1 / f ** 0.5

    # Nikuradse - Smooth
    elif model == 'niks':
        fun = 2 * np.log10(Re*f**0.5) - 1 / f ** 0.5

    # Samadianfard
    elif model == 'sam':
        fun = (Re**(eta/dh) - 0.6315093)/(Re**(1/3.0) + Re*(eta/dh)) + 0.0275308*(6.929841/Re + (10**(eta/dh)/(eta/dh + 4.781616)))*((eta/dh)**0.5+9.99701/Re)

    # Diaz-Plascencia
    elif model == 'diaz':
        lam2 = abs(0.02 - (1.0/(-2*np.log10(eta/(3.7*dh))))**2)
        tau2 = 0.77505/(eta/dh)**2 - 10.984/(eta/dh) + 7953.89
        fun = 64/Re + 0.02/(1 + np.exp((3000 - Re)/100)) + lam2/(1 + np.exp((eta/dh)*(tau2 - Re)/150))

    # Afzal
    elif model == 'afzal':
        Rs = reynolds_roughness(Re, f, eta)
        fun = -1.93*np.log10((1.9/(Re*f**0.5))*(1+0.34*Rs*np.exp(-11/Rs))) - 1/f**0.5

    return fun


def friction_factor(Re, eta, dh=d, model='cole'):
    # Mode Dict
    modes = {
        'cole': 'imp',
        'nikr': 'exp',
        'niks': 'exp',
        'sam': 'exp',
        'diaz': 'exp',
        'afzal': 'imp'
    }
    mode = modes[model]

    if mode == 'imp':
        f_r = newton(friction_model, args=(Re, eta, model), x0=0.002, maxiter=10000)
    else:
        f_r = friction_model(0, Re, eta, model=model)
    return f_r


def minimizer(eta, model='cole', obj=1):
    # Mode Dict
    modes = {
        'cole': 'imp',
        'nikr': 'exp',
        'niks': 'exp',
        'sam': 'exp',
        'diaz': 'exp',
        'afzal': 'imp'
    }
    mode = modes[model]

    # Utilize Newton-Raphson Method to Calculate Friction Factor
    fs = np.zeros(len(Res))
    for i in range(len(Res)):
        if mode=='imp':
            fs[i] = newton(friction_model, args=(Res[i], eta), x0=0.002, maxiter=10000)
        else:
            fs[i] = friction_model(0, Res[i], eta)

    if obj == 1:
        obj = np.sum((fs - Cf)**2)
    elif obj == 2:
        obj = max(fs-Cf) - min(fs-Cf)
    elif obj == 3:
        obj = max(abs(fs - Cf)) - min(abs(fs - Cf))
    elif obj == 4:
        obj = 1/(np.mean(abs(fs-Cf))/np.std(abs(fs-Cf)))

    return obj


res = minimize(minimizer, x0=0.00001, bounds=((1e-7, 1e-3),), args=('afzal', 1))
eta_calc = res.x[0]

print(res)

print('Calculated Surface Roughness: {0}mm'.format(1000*eta_calc))

Re_fit = np.linspace(0.5*min(Res), 5*max(Res), 100)
Cf_fit = [friction_factor(i, eta_calc, model='afzal') for i in Re_fit]

# Plot the Moody Diagram
roughness = np.array(
    [0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4,
     10**-4, 5e-5, 10**-5, 5e-6, 10**-6, 7.4626e-5]
)
plot_Res = np.geomspace(4000, 10**8, 100)
friction_result = np.zeros(len(plot_Res))
friction_data = {}
for i, e in enumerate(roughness):
    res = friction_result.copy()
    # Loop through each data point in calculate the friction
    for j, r in enumerate(plot_Res):
        res[j] = friction_factor(r, d*e, model='cole')

    friction_data.update({str(e): res})

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(9, 6.5), sharey='row', layout='constrained')

# Titles
fig.suptitle('Colebrook-White Moody Diagram', fontsize=11)

# Styles
from matplotlib import font_manager

font_path = 'EBGaramond-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

for key in friction_data.keys():
    c = 'red' if str(7.4626e-5) == key else '#77C3EC'
    label = 'PVC Pipe' if str(7.4626e-5) == key else None
    axs.plot(plot_Res, friction_data[key], c=c, linewidth=1.0, label=label)

# Plot Experimental Data
axs.plot(Res, Cf, linewidth=1,
               color='black',
               label='Experimental Data '+r'$\frac{\epsilon}{D_H} = $'+'{0}'.format(round(eta_calc/d, 4)))

# Parasite Axis
axst = axs.twinx()

# Styling
axs.grid(which='both', linestyle='dashed', linewidth=0.5)
axs.set_xlabel('Re')
axs.set_ylabel('Darcy-Weisbach Friction Factor')
axst.set_ylabel('Relative Roughness '+r'$\frac{\epsilon}{D_H}$')

# Relative Roughness Ticks
e_tick_labels = friction_data.keys()
e_tick_locs = [friction_data[k][-1] for k in e_tick_labels]
# Axis Scaling
ax_ylim = axs.get_ylim()
#y1 = ax_ylim[0]/dict_limit(friction_data, np.min)
#y2 = ax_ylim[1]/dict_limit(friction_data, np.max)
#print(y1, y2)

# Formatting
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlim(axs.get_xlim()[0], max(plot_Res))
axs.yaxis.set_major_formatter(ScalarFormatter())
axs.yaxis.set_major_locator(MaxNLocator(10))
axst.set_yscale('log')
axst.set_ylim(axs.get_ylim())
axst.set_yticks(ticks=e_tick_locs, labels=e_tick_labels)

# Legend
handles, labels = axs.get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower right')
fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.1, 0.1), bbox_transform=plt.gcf().transFigure)

#plt.savefig('charts/colebrook_moody.png', dpi=400)

plt.show()

#eps = 0.009/1000
#plt.plot(reynolds_roughness(Res, Cf, eta_calc), roughness_function(Cf, eta_calc))
#plt.plot(reynolds_roughness(Re_fit, Cf_fit, eta_calc), roughness_function(Cf_fit, eta_calc))
#plt.show()

#plt.plot(Res, Cf)
#plt.plot(Re_fit, Cf_fit)
#plt.xscale('log')
#plt.yscale('log')
#plt.show()