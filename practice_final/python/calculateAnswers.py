#%%
import sympy as sp
from sympy.physics.vector import dynamicsymbols, kinematic_equations
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display


from sympy import sin, cos, diff, Matrix, symbols, simplify, init_printing
init_printing()

def dotprint(expr):
    display(Math(vlatex(expr)))

#%%
# Defining mathematical variables (called symbols in sympy) and time varying 
# functions like z and theta and h
# t, mc, mr, Jc, d, mu, g, F, tau, fl, fr = symbols('t, m_c, m_r, Jc, d, mu, g, F, tau, f_l, f_r')
t, m, ell, g, k1, k2, tau, b, Jc, P_0 = symbols('t, m, ell, g, k_1, k_2, tau, b, Jc, P_0')
theta = dynamicsymbols('theta')
thetad = theta.diff(t)
thetadd = thetad.diff(t)

#%%
# Defining vectors of generalized coordinates and their derivatives
q = Matrix([[theta]])
q_dot = q.diff(t)

#%%
# Defining the position of each mass, and then finding its velocity
p = Matrix([[ell*theta]])

v = diff(p, t)

#%%
# Defining angular velocity and rotational inertial matrix about the center pod
omega = Matrix([[thetad]])
J = Matrix([[Jc]])


# %% [markdown]
# # Find kinetic energy:
K = simplify(0.5*m*v.T*v)
dotprint(K)

# just grabbing the scalar inside this matrix so that we can do L = K-P, since P is a scalar
K = K[0,0]
#dotprint(theta**2)

# %% [markdown]
# # Defining potential energy:
P = simplify(m*g*(ell*sin(theta)) + 0.5*k1*theta**2 +0.25*k2*theta**4 + P_0)
dotprint(P)

#%%
# Calculate the lagrangian, using simplify intermittently can help the equations to be
# simpler, there are also options for factoring and grouping if you look at the sympy
# documentation.
L = simplify(K-P)
dotprint(L)

#%%
# Now find generalized forces and torques
#force = Matrix([[-(fr+fl)*sin(theta)], [(fr+fl)*cos(theta)], [d*(fr-fl)]])
force = Matrix([[tau]])
drag = Matrix([[-b*thetad]])
RHS = force + drag

#%% [markdown]
# # Find the equations of motion by computing
# $\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}} \right) - \frac{\partial L}{\partial q} = \tau - B\dot{q}$

LHS = simplify( diff(diff(L, q_dot), t) - diff(L, q) )

dotprint(sp.Eq(LHS, RHS))

full_eom = LHS - RHS

#%%
# Now solve for the accelerations, zdd, hdd, and thetadd
result = simplify(sp.solve(full_eom, (thetadd)))

# Result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
thetadd_eom = result[thetadd] # EOM for thetadd, as a function of states and inputs

dotprint(thetadd_eom)

#%%
############################################################
### Defining vectors for x_dot, x, and u, then taking partial derivatives
############################################################

# defining states and inputs symbolically
state_variable_form = Matrix([[thetad], [thetadd_eom]])
states = Matrix([[theta], [thetad]])
inputs = Matrix([[tau]])

#%%
# finding the jacobian with respect to states (A) and inputs (B)
A = state_variable_form.jacobian(states)
B = state_variable_form.jacobian(inputs)

# substituting in the equilibrium values for each state and input (finding the 
# equilibrium points can likely be done automatically in sympy as well, but 
# we are currently defining them by hand)
A_lin = A.subs([(theta.diff(t),0), (theta, 0), (tau, 0)])
B_lin = B.subs([(theta.diff(t),0), (theta, 0), (tau, 0)])

dotprint(A_lin)
dotprint(B_lin)

#%%
