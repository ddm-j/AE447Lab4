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

# Reynolds Calculation
Res = rho*vel*d/mu

# Friction Factor Calculation
Cf = data[:, 1]/(0.5*rho*(l/d)*data[:, 0]**2)

# Log Law Boundary Layer Calculation
tau = 4*Cf*0.5*rho*vel**2
eps = 0.01/(10*1000)
v_f = np.sqrt(tau/rho)
Rew = v_f*eps/(mu/rho)
print(Rew)

# Solution of the Boundary layer equation
u, ut, k, nu, y, y0 = sym.symbols("u, u_t, k, nu, y, y_0")
eq = (ut/k)*sym.log(y/y0) - u
eq = eq.subs(y0, nu/(9*ut)).simplify()
ut_sol = sym.solve(eq, ut)[0]
ut_sol = ut_sol.subs(
    {
        nu: mu/rho,
        k: 0.41,
        y: d/2,
    }
)
#ut_f = sym.lambdify(ut_sol, u)

for i in range(len(vel)):
    y_calc = np.geomspace(1e-6, d/2, 100)
    ut_calc = float(ut_sol.subs(u, vel[i]))
    print("Comparison of Analytical & Experimental: {0}, {1}".format(ut_calc, np.sqrt(tau[i]/rho)))
    print(np.sqrt(tau[i]/rho), np.sqrt((4*Cf[i]*0.5*rho*vel[i]**2)/rho))
    #print(eps*ut_calc/(mu/rho))
    u_calc = (ut_calc/0.41)*np.log(y_calc/((mu/rho)/(9*ut_calc)))

    plt.plot(y_calc, u_calc)
plt.show()