# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:35:04 2021
refer to https://www.youtube.com/watch?v=ay0zZ8SUMSk
"""
# QUANTUM MECHANICS
# Eigenstates of any 1D potential 
"""
The 1D Schrodinger equation was converted into an eigenvalue function. Specifically, it takes the following form:

| (1/dy)+mL^2V1    -1/2dy^2         0        0 . .    || psi1 |        |psi1
|                                                     ||      |        |
|  -1/2dy^2     (1/dy)+mL^2V2    -1/2dy^2    0 . .    || psi2 |        |psi2
|                                                     ||      | = mL^2E|
|   . . .             . . .       . . .    -1/2dy^2   || ...  |        |...
|                                                     ||      |        |
|   . . 0             . . 0           (1/dy)+mL^2Vn-1 || psi_n|        |psi_n

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

N = 2000
dy = 1/N
y = np.linspace(0, 1, N+1)

def mL2V(y):
    return 1000*(y-1/2)**2
V = mL2V(y)

d = 1/(dy**2) + mL2V(y)[1:-1]               # setting up the main diagonal
e = -1/(2*dy**2) * np.ones(len(d)-1)        # setting up the outer diagonals
# note that the np.ones is used to fill up the array

w, v = eigh_tridiagonal(d, e)           # w = eigenenergies, v = eigenfunct


# to transpose a matrix, use the call .T
# uncomment to plot the eigenfunctions 
"""
plt.plot(v.T[0])
plt.plot(v.T[1])
plt.plot(v.T[2])
plt.plot(v.T[3])
"""
# uncomment to plot probability density
 
plt.plot(v.T[0]**2)
plt.plot(v.T[1]**2)
plt.plot(v.T[2]**2)
plt.plot(v.T[3]**2)

# creating a bar plot for the eigenenrgies (eigenvalues)
#plt.bar(np.arange(0, 10, 1), w[0:10])
#plt.ylabel('$mL^2 E/\hbar^2$', fontsize = 15)

