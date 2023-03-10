'''
Filename: pattern_generation.py
Author: Matthew Peres Tino
Description: Holds functions for pattern generation

'''

from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np

def genFourierPattern(sym, lamb, theta=0):
    """
    Generates prototypical patterns using two-dimensional Fourier series [ref]
    [ref] = G~H Gunaratne et al., "Pattern formation in the presence of symmetries", 1994
    
    Parameters
    ----------
    sym : int
       Dominant m-fold symmetry of pattern. I.e., stripe pattern m=2. Hexagonal m=6, square m=4...
    lamb : float
        The wavelength of the pattern (unitless).
    theta : int or float
        The angle (in degrees) of which we wish to CCW rotate the pattern.
        Default = 0 degrees.
    
    Returns
    -------
    f : function
        A bivariate function f(x,y) that will generate the patterned
        image specified.
    """
    # Define basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    # if we need to rotate basis vectors, apply rotation matrix in 2D euclidean space
    if theta != 0:
        theta *= np.pi / 180 # convert to radian
        rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        e1 = np.dot(rotmat,e1)
        e2 = np.dot(rotmat,e2) 

    # now we need to handle the k vectors
    if sym == 2 or sym == 6:
        k1 = (2 * np.pi / lamb) * e2
        k2 = (2 * np.pi / lamb) * ((np.sqrt(3) / 2) * e1 - 0.5 * e2)
        k3 = (2 * np.pi / lamb) * (-(np.sqrt(3) / 2) * e1 - 0.5 * e2)
        if sym == 2:
            a1, a2, a3 = 1, 0, 0
        elif sym == 6:
            a1, a2, a3 = 1, 1, 1
    elif sym == 4:
        k1 = (2 * np.pi / lamb) * e1
        k2 = (2 * np.pi / lamb) * e2
        k3 = np.array([0,0]) # dumpy vector, not needed for m=4 symmetry
        a1, a2, a3 = 1, 1, 0

    # Dot function for 2d as np.dot is not compatible with meshes.
    dot2d = lambda n,m: n[0] * m[0] + n[1] * m[1]
    # Define the function and return. Note np.dot is not compatible with meshes.
    f = lambda x, y: a1 * np.exp(1j * dot2d(k1, np.array([x, y]))) \
                   + a2 * np.exp(1j * dot2d(k2, np.array([x, y]))) \
                   + a3 * np.exp(1j * dot2d(k3, np.array([x, y])))
    return(f)

def genMesh(F, x, y, Nx, Ny):
    """
    Generates a rectangular mesh that discretizes a given function.
    Essentially this is used to project a continuous function onto a
    discrete domain.
    
    Parameters
    ----------
    F : function
        The function to project.
    x : list
        The bounds for the first variable. Example: [0, 1]
    y : list
        The bounds for the second variable.
    Nx : int
        The number of points to discretize x by.
    Ny : int
        The number of points to discretize y by.
    
    Returns
    -------
    X : np.ndarray
        The grid for the first variable.
    Y : np.ndarray
        The grid for the second variable.
    M : np.ndarray
        An Nx*Ny matrix for F projected onto a discrete domain.
    """
    # Create variable space
    x_rng = np.linspace(x[0], x[1], Nx)
    y_rng = np.linspace(y[0], y[1], Ny)
    X,Y = np.meshgrid(x_rng, y_rng)
    
    #return (X, Y, F(X,Y))
    return F(X,Y)

if __name__ == "__main__":
    pattern = genFourierPattern(sym=2, lamb=30,theta=0)
    N = 250
    discPattern = np.real(genMesh(pattern, x = [-(N-1)/2.0, (N-1)/2.0], y = [-(N-1)/2.0, (N-1)/2.0], Nx = N, Ny = N))
    plt.imshow(discPattern,cmap='gray')
    plt.show()
