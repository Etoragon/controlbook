import matplotlib.pyplot as plt
import rodMassParam as P
from rodMassAnimation import rodMassAnimation
from dataPlotter import dataPlotter
from rodMassDynamics import rodMassDynamics
import numpy as np
from signalGenerator import signalGenerator

# instantiate arm, controller, and reference classes
rodMass = rodMassDynamics()

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = rodMassAnimation()
# control gain calculations go here
th_eq = signalGenerator(amplitude=np.pi/2)
tau_eq = 0

def controller(theta):
    firstTerm = P.ell * P.g * P.m * np.cos(theta)
    secondTerm = P.k1 * theta 
    thirdTerm = P.k2 * (theta ** 3 )

    print("-------------------")
    print(firstTerm)
    print(secondTerm + thirdTerm)
    return firstTerm + secondTerm + thirdTerm

t = P.t_start
while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        u = controller(th_eq.step(t-3))
        y = rodMass.update(u)
        print("updated: " + str(y))
        t += P.Ts
    # update animation and data plots
    animation.update(rodMass.state)
    dataPlot.update(t, th_eq.step(t-3), rodMass.state, u)
    plt.pause(0.01)

# Keeps the program from closing until the user presses a button.d
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
