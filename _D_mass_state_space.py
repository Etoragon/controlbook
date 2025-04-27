#%%
import numpy as np
import control as ctrl
from control import *
from control.matlab import *
import sympy as sp
from sympy import *
import matplotlib.pyplot as plt
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display
#sin, cos, diff, Matrix, symbols, simplify, init_printing
init_printing()

def dotprint(expr):
    display(Math(vlatex(expr)))

# %% Define symbols
# Let's find the state space form from scratch
t, m, k, b, F = symbols('t m k b F')
z = dynamicsymbols('z')

#%% Find the position, velocity
# Generalized coordinates
q = Matrix([[z]])
q_dot = q.diff(t)

# Position
p = Matrix([[z], [0], [0]])
# Velocity
v = diff(p, t)

#%% Generalized forces and torques for RHS
friction = Matrix([b*diff(z,t)])
RHS = simplify(Matrix([F]) - friction)

#%% Kinetic and Potential energy, and Lagrangian for LHS
KineticE = simplify(0.5*m*v.T*v)
KineticE = KineticE[0]
PotentialE = simplify(0.5*k*(z**2))
L = simplify(KineticE - PotentialE)
LHS = simplify(diff(diff(L, q_dot), t) - diff(L, q))

#%% Full EOMs and simplification
eoms = simplify(LHS-RHS)
#eoms = eoms[0]
zdot = z.diff(t)
zddot = zdot.diff(t)
result = simplify(sp.solve(eoms, (zddot)))

#%% State space form
#state_var_form = Matrix([eoms])
state_var_form = Matrix([[zdot], [result[zddot]]])
states = Matrix([[z], [zdot]])
inputs = Matrix([[F]])
A = state_var_form.jacobian(states)
B = state_var_form.jacobian(inputs)

#%% Get SS equations
# Physical parameters of the SMD
mass = 5.0 # mass kg
k_spring = 3.0  # spring constant Kg/s^2
b_damp = 0.5  # damping coefficient Kg/s
# State Space Equations
# xdot = A*x + B*u
# y = C*x
A = np.array([[0.0, 1.0],
            [-k_spring / mass, -b_damp / mass]])
B = np.array([[0.0],
            [1.0 / mass]])
C = np.array([[1.0, 0.0]])

# Initial Conditions
x0 = 0.0  # initial position of mass, m
xdot0 = 0.0  # initial velocity of mass m/s

# Simulation Parameters
t_start = 0.0 # Start time of simulation
t_end = 50.0  # End time of simulation
T = np.linspace(t_start, t_end, 1000) # time vector
#Ts = 0.01  # sample time for simulation
input = 1.0 # step

#%% Control design
tr = 2.0
zeta = 0.707
# gain calculation
wn = 2.2/tr  # natural frequency
des_char_poly = [1, 2*zeta*wn, wn**2]
des_poles = np.roots(des_char_poly)
# Compute the gains if the system is controllable
if np.linalg.matrix_rank(ctrl.ctrb(A, B)) != 2:
    print("The system is not controllable")
else:
    K = ctrl.acker(A, B, des_poles) # or if MIMO
    K = ctrl.place(A, B, des_poles)
    kr = -1.0/(C @ np.linalg.inv(A - B @ K) @ B)
    kr = 1.0 # what if there was no feedforward?
print('K: ', K)
print('kr: ', kr)

# %% Closed loop response
sys = ctrl.ss((A-B@K), B*kr, C, 0)
t, y = ctrl.forced_response(sys, T, input, X0=[x0,xdot0])

# %% Plot
plt.figure()
plt.plot(t, y, 'b-')
#plt.legend(['$x$', '$x_2$'])
#plt.xlim(0, 50)
plt.ylabel('Position [m]')
plt.xlabel('Time [s]')
plt.title("Step response from $x = 0$");

# %% Let's add some integrator states
Cr = np.array([1.0, 0.0]) 
# We have only one integrator state
# Form the augmented system
Aaug = np.block([
    [A, np.zeros((2,1))],
    [-Cr, np.zeros((1,1))]
])
Baug = np.block([
    [B],
    [np.zeros((1,1))]
])
Br = np.block([
    [np.zeros((2,1))],
    [np.eye(1)]
])
Caug = np.array([[1.0, 0., 0.]])

# %% Form the desired char polynomial
tr = 2.0
tr = 2.5
zeta = 0.707
zeta = 0.95
integrator_pole = -1.0
integrator_pole = -1.2
# NOTE: if zeta is NOT 0.707 then rise time is half the peak time
# tp = np.pi/(wn*np.sqrt(1-zeta**2)) # From Ch 8
wn = np.pi/(2*tr*np.sqrt(1-zeta**2))
des_char_poly = np.convolve([1, 2*zeta*wn, wn**2],
                            [1, -integrator_pole])
des_poles = np.roots(des_char_poly)

# %% Check controllability
if np.linalg.matrix_rank(ctrl.ctrb(Aaug, Baug)) != 3:
    print("The system is not controllable")

Kaug = ctrl.place(Aaug, Baug, des_poles) # use place
KI = Kaug[0,2]
K = Kaug[0,0:2]
print('Kaug: ', Kaug)
print('K: ', K)
print('Ki: ', KI)

# %% Closed loop response
sys = ctrl.ss((Aaug-Baug@Kaug), Br, Caug, 0)
t, y = ctrl.forced_response(sys, T, input, X0=[x0,xdot0,0.0])

# %% Plot
plt.figure()
plt.plot(t, y, 'b-')
#plt.legend(['$x$', '$x_2$'])
#plt.xlim(0, 50)
plt.ylabel('Position [m]')
plt.xlabel('Time [s]')
plt.title("Step response from $x = 0$ with Integrator States");

# %%
plt.show() # shows the plot when run from a terminal
