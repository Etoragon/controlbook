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
t, mc, mr, Jc, d, mu, g, F, tau, fl, fr = symbols('t, m_c, m_r, Jc, d, mu, g, F, tau, f_l, f_r')
z = dynamicsymbols('z')
zd = z.diff(t)
zdd = zd.diff(t)
h = dynamicsymbols('h')
hd = h.diff(t)
hdd = hd.diff(t)
theta = dynamicsymbols('theta')
thetad = theta.diff(t)
thetadd = thetad.diff(t)

#%%
# Defining vectors of generalized coordinates and their derivatives
q = Matrix([[z], [h], [theta]])
q_dot = q.diff(t)

#%%
# Defining the position of each mass, and then finding its velocity
pc = Matrix([[z], [h], [0]])
pr = Matrix([[z + d*cos(theta)], [h+d*sin(theta)], [0]])
pl = Matrix([[z - d*cos(theta)], [h-d*sin(theta)], [0]])

vc = diff(pc, t)
vr = diff(pr, t)
vl = diff(pl, t)

#%%
# Defining angular velocity and rotational inertial matrix about the center pod
omega = Matrix([[0], [0], [thetad]])
J = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, Jc]])


# %% [markdown]
# # Find kinetic energy:
K = simplify(0.5*mc*vc.T*vc + 0.5*mr*vr.T*vr + 0.5*mr*vl.T*vl + 0.5*omega.T*J*omega)
dotprint(K)

# just grabbing the scalar inside this matrix so that we can do L = K-P, since P is a scalar
K = K[0,0]

# %% [markdown]
# # Defining potential energy:
P = simplify(mc*g*h + mr*g*(h+d*sin(theta)) + mr*g*(h-d*sin(theta)))
dotprint(P)

#%%
# Calculate the lagrangian, using simplify intermittently can help the equations to be
# simpler, there are also options for factoring and grouping if you look at the sympy
# documentation.
L = simplify(K-P)

#%%
# Now find generalized forces and torques
#force = Matrix([[-(fr+fl)*sin(theta)], [(fr+fl)*cos(theta)], [d*(fr-fl)]])
force = Matrix([[-F*sin(theta)], [F*cos(theta)], [tau]])
drag = Matrix([[-mu*zd], [0], [0]])
RHS = force + drag

#%% [markdown]
# # Find the equations of motion by computing
# $\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}} \right) - \frac{\partial L}{\partial q} = \tau - B\dot{q}$

LHS = simplify( diff(diff(L, q_dot), t) - diff(L, q) )

dotprint(sp.Eq(LHS, RHS))

full_eom = LHS - RHS

#%%
# Now solve for the accelerations, zdd, hdd, and thetadd
result = simplify(sp.solve(full_eom, (zdd, hdd, thetadd)))

# Result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
zdd_eom = result[zdd]  # EOM for zdd, as a function of states and inputs
hdd_eom = result[hdd]
thetadd_eom = result[thetadd] # EOM for thetadd, as a function of states and inputs

dotprint(zdd_eom)
dotprint(hdd_eom)
dotprint(thetadd_eom)

#%%
############################################################
### Defining vectors for x_dot, x, and u, then taking partial derivatives
############################################################

# defining states and inputs symbolically
state_variable_form = Matrix([[zd], [zdd_eom], [hd], [hdd_eom], [thetad], [thetadd_eom]])
states = Matrix([[z], [zd], [h], [hd], [theta], [thetad]])
inputs = Matrix([[F], [tau]])

#%%
# finding the jacobian with respect to states (A) and inputs (B)
A = state_variable_form.jacobian(states)
B = state_variable_form.jacobian(inputs)

# substituting in the equilibrium values for each state and input (finding the 
# equilibrium points can likely be done automatically in sympy as well, but 
# we are currently defining them by hand)
A_lin = A.subs([(theta.diff(t),0), (theta, 0), (F, (mc+2*mr)*g), (tau, 0)])
B_lin = B.subs([(theta.diff(t),0), (theta, 0), (F, (mc+2*mr)*g), (tau, 0)])

dotprint(A_lin)
dotprint(B_lin)

#%%
