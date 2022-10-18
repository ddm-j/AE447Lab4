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


# Layer Function
def turb_bl(d, v_f, n, points=100):
    # Discretize the Sections:
    y_visc = list(np.linspace(0.0, 5 * n / v_f, points))
    t_buff = list(np.linspace(0.0, 1.0, points))
    y_log = list(np.geomspace(30 * n / v_f, d, points))

    # Calculate the profile
    u_visc = []
    u_buff_bez = []
    y_buff_bez = []
    u_log = []
    for i in range(0, len(y_visc)):
        # Viscous Sublayer
        u_visc.append(u_v.subs({y: y_visc[i], ut: v_f, nu: n}))

        # Buffer Layer
        # Convert to Dimensionless Wall Parameter
        up_b = Bz.subs({
            yp1: 5,
            yp2: 30,
            C: 5.0,
            k: 0.41,
            t: t_buff[i]
        }).evalf()
        u_buff_bez.append(float(up_b[1, 0]) * v_f)
        y_buff_bez.append(float(up_b[0, 0]) * n / v_f)

        # Log Law Layer
        u_log.append(v_f * float(u_l.subs({y: y_log[i] * v_f / n, k: 0.41})))

    # Total Velocity Profile
    y_tot = y_visc + y_buff_bez + y_log
    u_tot = u_visc + u_buff_bez + u_log

    # Results Array
    res = np.zeros((len(y_tot), 2))
    res[:, 0] = y_tot
    res[:, 1] = u_tot

    # Plot
    #axs.plot([i * v_f / n for i in y_visc], [i * v_f for i in u_visc], label='Viscous Sublayer')
    #axs.plot([i * v_f / n for i in y_buff_bez], [i * v_f for i in u_buff_bez], label='Cubic Buffer Layer')
    #axs.plot([i * v_f / n for i in y_buff_cub], [i * v_f for i in u_buff_cub], label='Bezier Buffer Layer')
    #axs.plot([i * v_f / n for i in y_log[:30]], [i * v_f for i in u_log[:30]], label='Log-Law Layer')
    #axs.xscale('log')
    #axs.xlabel(r'$y^+$')
    #axs.ylabel(r'$u^+$')
    #axs.legend()

    return res

boundary_data = {}
thickness_data = {}
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

    # Calculate the Boundary Layer
    boundary_data.update(
        {
            str(vel[i]): turb_bl(d/2, ut_calc, mu/rho)
        }
    )

    # Calculate the Boundary Layer Thickness
    delt = ((mu/rho)/ut_calc)*np.exp((0.41/ut_calc)*(vel[i]-5.0*ut_calc))
    u_delt = (ut_calc/0.41)*np.log(delt*ut_calc/(mu/rho)) + 5.0*ut_calc
    print(delt, u_delt)
    thickness_data.update(
        {
            str(vel[i]): [delt, u_delt]
        }
    )


# Plotting
fig, axs = plt.subplots(1, 1, figsize=(6.5, 4), sharey='row', layout='constrained')

# Titles
fig.suptitle('Turbulent Boundary Layer Profile', fontsize=11)

# Styles
from matplotlib import font_manager

font_path = 'EBGaramond-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

for i, key in enumerate(boundary_data.keys()):
    y_set = boundary_data[key][:, 0]
    u_set = boundary_data[key][:, 1]

    if i == 0:
        print(u_set/float(key))
    # Plot Boundary Layer
    #label=r'$U_\infty = {0}$'.format(key),
    axs.plot(y_set, u_set,
                       linewidth=1,
                       color='black')

    # Plot U99 thickness point
    if i == len(boundary_data)-1:
        label = r'$U_{99}, \delta_t$'
    else:
        label = None
    axs.scatter([thickness_data[key][0]], [thickness_data[key][1]],
             marker='o', s=20, color='black', label=label)
# Pipe Axis
axs.axvline(d/2, linestyle='dashed', linewidth=0.75, color='black')

# Formatting
axs.set_ylabel(r'$u(y)\quad \frac{m}{s}$')
axs.set_xlabel(r'$y\quad (m)$')

# Parasite Axis
axst = axs.twinx()
axst.set_ylim(axs.get_ylim())
axst.set_ylabel('Tested Velocities')
vel_labs = thickness_data.keys()
vel_locs = [np.max(boundary_data[k][:, 1]) for k in vel_labs]
print(vel_labs)
print(vel_locs)
axst.set_yticks(ticks=vel_locs, labels = vel_labs)

handles, labels = axs.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=plt.gcf().transFigure)
plt.show()
