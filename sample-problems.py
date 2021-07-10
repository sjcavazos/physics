# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:25:30 2021

@author: sjcav
"""
import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1: PHYSICS 101
"""
The equation for the height of a thrown ball is

y = -1/2gt^2 + v0t

y - height of the ball
g - gravity
t - time
v0 - initial upward velocity

P1 throws a ball at 10 m/s while P2 throws a ball at 15 m/s. Plot each trajectory.

"""

t = np.linspace(0,2, 100)
g = 9.81
v0_P1 = 10.0
v0_P2 = 15.0

y_P1 = -1/2*g*t**2 + v0_P1*t

plt.plot(t,y_P1, 'r', label='P01')

y_P2 = -1/2*g*t**2 + v0_P2*t 

plt.plot(t,y_P2, 'g', label='P02')
plt.xlabel('time', fontsize=15)
plt.ylabel('height', fontsize=15)
plt.grid()
plt.legend()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# PROBLEM 2: POETRY

"""
The poem below is organized in such a way that each poem line is a new element in a list
"""
poem = ['when the day comes we ask ourselves,',
        'where can we find light in this never-ending shade?',
        'the loss we carry,',
        'a sea we must wade,',
        "we've braved the belly of the beast,",
        "we've learned that quiet isn't always peace,",
        'and the norms and notions,',
        'of what just is']
"""
Loop through the lines of the poem and if the line contains the substring 'we', then print 'line ___ contains we', where '___' is the line number.

"""
print(poem[1])

for i, line in enumerate(poem):
    if 'we' in line:
        print('Line {} contains we'.format(i))

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# PROBLEM 3: A COUNTING PROBLEM

"""
Add up every number from 1 to 999999, except those that can be divided by 4, and those that can be divided by 6.
"""

summation = 0
for i in range(1000000):
    if not(i%4==0) and not(i%6==0):
        summation = summation + i

