# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:28:16 2021
refer to https://www.youtube.com/watch?v=DF1SnjXZcbM&t=62s
"""
# QUANTUM MECHANICS
# 2D Schrodinger Equation
"""
The 2D Schrodinger equation is discretized and turned into a matrix, just like the 1D problem. Recall the identity for discretizing a second derivative!
Specifically, the 2DSE takes the following form:
    
1(psi_i-1) + (2m*dx^2*-Vi-2)*(psi_i) + 1(psi_i-1) = 2m*dx^2*E*(psi_i)

let di = -2 + 2m*dx^2*Vi

| d1    1    0    0    ..0 || psi1 |          | psi1  |
|                          ||      |          |       |
|  1    d2   1    0    ..0 || psi2 |          | psi2  |
|                          ||      | = 2m*dx*E|       |
|  0    1    d3   1    ..0 || ...  |          | ...   |
|                          ||      |          |       |
| ...        1    dn     0 || psi_n|          | psi_n |

During the process of numerically solving the 2DSE, we will need to stack the N-dimensional grid into a vector before it can be solved.

| y11   y12   y13 |        | y11 |
|                 |        | y12 |
| y21   y22   y23 | ---->  | y13 |
|                 |        | ... |
| y31   y32   y33 |        | y33 |

Recall the Kronecker Product to obtain a second derivative inside a matrix... More of this is referenced in the video.

[-1/2(D(+)D) + m*dx^2*V]*(psi) = (m*dx^2*E)*(psi) 

"""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
#plt.style.use(['science','notebook'])
from scipy import sparse
# ---                                            ---

# create a meshgrid of x and y coordinates
N = 150
X , Y = np.meshgrid(np.linspace(0,1,N, dtype=float),
                    np.linspace(0,1,N, dtype=float))

# this returns m*dx*x^2*V
def get_potential(x,y):
    return 0*x
V = get_potential(X,Y)

# now we construct the diagonals, -1/2(D(+)D) + m*dx^2*V]

"""
Recall that our Diagonals and matrices are defined as the following:

      | -2   1   0    0 |
 D =  | 1   -2   1    0 |
      | 0   1   -2    1 |
      |     0    1   -2 |
 
 T= D (+) D

      | V11   V12   V1N |
      |                 |
 V =  | V21   V22   V2N |
      |                 |
      | ...   ...   VNN |
      
      | V11   V12   V1N |
      |                 |
 U =  | V21   V22   V2N |
      |                 |
      | ...   ...   VNN |      

 
"""

# Constructing the sparse matrices
diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = -1/2 * sparse.kronsum(D,D)
U = sparse.diags(V.reshape(N**2), (0))
H = T + U

eigenvalues, eigenvectors = eigsh(H, k=10, which='SM')
print(eigenvalues)

"""
Note: before plotting, we need to reshape our arrays. The eigenvalues are on a NxN grid, but we want to plot it in our 2D space. Therefore, we now do a bit of matrix re-arranging:
"""
# Plotting the eigenvectors
def get_e(n):
    return eigenvectors.T[n].reshape((N,N))
# get_e(n) now gives you the energy levels!

"""
plt.figure(figsize=(9,9))
plt.contourf(X, Y, get_e(4)**2, 30)              # 30 contour levels
"""

# 3D Animation
my_cmap = plt.get_cmap('cool')
def init():
    # plot the surface
    ax.plot_surface(X, Y, get_e(4)**2, cmap=my_cmap,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    ax.set_zlabel('$\propto|\psi|^2$')
    return fig,

def animate(i):
    ax.view_init(elev=10, azim=4*i)
    return fig,

fig = plt.figure()
ax = Axes3D(fig)
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=90, interval=50)
ani.save('rotate_azimuth_angle_3_surf.gif', writer='pillow',fps=20)


# Note that for Quantum Mechanics, we are usually more interested in the energies (at least, in practice)

"""
For infinite square wells, the energies of the system are defined as the following:
    
    E_nx,ny = a(nx^2 + ny^2)

where the first energies are nx = ny = 1

This means we can find a = "half the lowest eigenvalue" and we can plot E/a which should be distributed like nx^2 + ny^2 for different combinations of nx and ny

alpha = eigenvalues[0]/2
E_div_alpha = eigenvalues/alpha
_ = np.arange(0, len(eigenvalues), 1)
plt.scatter(_, E_div_alpha)
"""



