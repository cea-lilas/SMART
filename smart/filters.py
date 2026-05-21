import numpy as np

def get_jarvis(theta):
    return lambda r: (r**2 / (4 * np.pi *(theta)**4)) * np.exp(-r**2 / (2. * (theta)**2))