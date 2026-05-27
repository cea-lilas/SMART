import numpy as np

def get_jarvis(theta):
    return lambda r: (r**2 / (4 * np.pi *(theta)**4)) * np.exp(-r**2 / (2. * (theta)**2))

def get_tanh(theta, x_c):
    return lambda r: 1/(1 + np.exp(6-150*r/theta) + np.exp(-47+50*(r/theta))) * np.tanh(r/(x_c*theta))/(r/(x_c*theta))

def get_miyazaki(theta):
    return lambda r: 1/(np.pi*theta**2) * (1 + (1 - r/theta)**2)*np.exp(r**2/(theta**2))