#%%
import sympy as sp
from sympy import Add
import control
import numpy as np
from sympy.physics.vector import dynamicsymbols, kinematic_equations
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display
import rodMassParam as params


from sympy import sin, cos, diff, Matrix, symbols, simplify, init_printing
init_printing()

def dotprint(expr):
    display(Math(vlatex(expr)))

#%%
# Defining mathematical variables (called symbols in sympy) and time varying 
# functions like z and theta and h
# t, mc, mr, Jc, d, mu, g, F, tau, fl, fr = symbols('t, m_c, m_r, Jc, d, mu, g, F, tau, f_l, f_r')
t, m, ell, g, k1, k2, tau, b, Jc, P_0 = symbols('t, m, ell, g, k_1, k_2, tau, b, Jc, P_0')
theta_e, tau_e = symbols('theta_e, tau_e')
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

#%% [markdown]
# # Part 1
# Question 1.1 Find tau_e such that theta_e is an equilibrium point, even when theta_e may not be zero

print()

####### Since our equilibrium point is where thetad = thetadd = 0, we can sub those values in
full_eom_lin = full_eom.subs([(theta.diff(t), 0), ((theta.diff(t)).diff(t), 0)])
full_eom_lin = simplify(full_eom_lin)

####### Now we will replace all theta with theta_e and tau with tau_e
full_eom_lin_symbols = full_eom_lin.subs([(theta, theta_e), (tau, tau_e)])

####### And finally solve for tau_e
final_equation = sp.solve(full_eom_lin_symbols, tau_e)

####### Print out new equilibrium equation
dotprint(final_equation)


#%% [markdown]
# Question 1.2 Create a controller that places a constant torque of tau_e on the physical system. 
print("This controller is given in the file as just a simple python function which accepts the reference theta and outputs the tau")



# %% [markdown]
# Question 1.3 Using Jacobian linearization, linearize the nonlinear model around the equilibrium (theta_e, tau_e)
# We will do this by first listing the EOM's
dotprint(full_eom)

# This function takes in a term, expands it to its Taylor series, then removes zeroes after taking the first two terms (theta^0 and theta^1, the constant and linear terms)
def linearizeThatTerm(term):
    print(term)
    dummy_theta = symbols("dummy_theta")
    
    # Replace only theta(t) -> dummy_theta, leave derivatives alone
    term_replaced = term.subs(theta, dummy_theta)
    
    # Perform Taylor series around dummy_theta = theta_e (i.e., around 0)
    linearized_term = term_replaced.series(dummy_theta, theta_e, 2).removeO()
    linearized_term_simplified = simplify(linearized_term)
    
    # Substitute dummy_theta back to theta
    linearized_term_corrected = linearized_term_simplified.subs(dummy_theta, theta)
    
    # Do not substitute tau -> tau_e unless you are linearizing control inputs separately
    return linearized_term_corrected


# Iterate through the full_eom and linearize each term
linearized_eom_terms = [linearizeThatTerm(term) for term in full_eom[0].args]
linearized_eom = Matrix([Add(*linearized_eom_terms)])



# Print the linearized equations of motion
dotprint(linearized_eom)



#%%
dotprint(full_eom)
# Now solve for the accelerations, zdd, hdd, and thetadd
result = simplify(sp.solve(full_eom, (thetadd)))
dotprint(result)

# Result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
thetadd_eom = result[thetadd] # EOM for thetadd, as a function of states and inputs

dotprint(thetadd_eom)

thetadd_eom_linearized_uncorrected = [linearizeThatTerm(term) for term in thetadd_eom.args]
thetadd_eom_linearized = Matrix([Add(*thetadd_eom_linearized_uncorrected)])
dotprint(thetadd_eom_linearized)

# %% [markdown]
# Question 1.4 Find transfer function when theta_e = 0
# We will do this by first listing the linearized EOM
dotprint(linearized_eom)


C = np.array([[1.0, 0.0]])

A_lin_valued = A_lin.subs([(k1, params.k1), (k2, params.k2), (b, params.b), (m, params.m), (ell, params.ell)])
B_lin_valued = B_lin.subs([(k1, params.k1), (k2, params.k2), (b, params.b), (m, params.m), (ell, params.ell)])

print(A_lin_valued)
print(B_lin_valued)

transferFunction = control.ss2tf(A_lin_valued, B_lin_valued, C, 0)

print(transferFunction)





# %%
# Question 1.5 Find state-space model linearized around equil params


dotprint(linearized_eom)


C = np.array([[1.0, 0.0]])

A_lin_valued = A_lin.subs([(k1, params.k1), (k2, params.k2), (b, params.b), (m, params.m), (ell, params.ell)])
B_lin_valued = B_lin.subs([(k1, params.k1), (k2, params.k2), (b, params.b), (m, params.m), (ell, params.ell)])

dotprint(A_lin_valued)
dotprint(B_lin_valued)
dotprint(C)
# %%
