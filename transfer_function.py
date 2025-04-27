#%%
from other_examples._F_planar_vtol.linearization import *
from sympy import eye, zeros, simplify

# The order for rows in C is based on the states I chose [z, zd, h, hd, theta, thetad]
C = Matrix([[1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0]])
D = Matrix(zeros(3,2))

#%%
# These are the three transfer functions for h, z, and theta with respect to inputs F and tau
s = symbols('s')
transfer_func = simplify(C@(s*eye(6)-A_lin).inv() @B_lin+D)

#%% [markdown]
# # Transfer Functions
# These indices that we select from transfer_func are based on the
# output (row) and input (column)\ 
# Output we chose: [z, h, theta] (based on C above)\ 
# Input we chose: [F, tau]
# [markdown]
# ### Full Transfer Function Matrix:
dotprint(transfer_func)
# [markdown]
# ### Individual Transfer Functions
print("\n\n\nTransfer function Z(s)/Tau(s)")
dotprint(transfer_func[0, 1])

print("\nTransfer function H(s)/F(s)")
dotprint(transfer_func[1, 0])

print("\n\n\nTransfer function Theta(s)/Tau(s)")
dotprint(transfer_func[2, 1])


# %%
