'''
Filename: pattern_generation.py
Author: Matthew Peres Tino
Description: Holds functions for pattern generation

'''

from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np

def rotateOriginalSize(s, angle):
    """
    Rotates an discrete grid with the return value the original size.
    
    Parameters
    ----------
    s : np.ndarray
        The image
    angle : float
        The angle to rotate by, in degrees.
    """
    
    s_rot = rotate(s, angle)
    # If the rotation changes the image shape, revert back to the original shape
    if s_rot.shape != s.shape:
        # Determine the padding on each side of the image caused by the rotation
        pad = s_rot.shape[0] - s.shape[0]
        # If 1 is padding then only one pixel is added and just remove last pixel
        if pad == 1:
            s_rot = s_rot[:-1, :-1]
        # If even padding, cast to int and trim
        elif pad % 2 == 0:
            pad = int(pad / 2)
            s_rot = s_rot[pad:-pad, pad:-pad]
        # If odd padding, remove last pixel, then amend pad, cast to int, and trim
        else:
            s_rot = s_rot[:-1, :-1]
            pad -= 1
            pad = int(pad / 2)
            s_rot = s_rot[pad:-pad, pad:-pad]
    
    return s_rot

def genFourierPattern(a, l):
    """
    Generates prototypical patterns using two-dimensional Fourier
    series.
    
    Parameters
    ----------
    a : list
       List of floats, [a1, a2, a3]
    l : float
        The wavelength of the pattern (unitless).
    
    Returns
    -------
    f : function
        A bivariate function f(x,y) that will generate the patterned
        image specified.
    """
    # Define basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    k1 = (2 * np.pi / l) * e2
    k2 = (2 * np.pi / l) * ((np.sqrt(3) / 2) * e1 - 0.5 * e2)
    k3 = (2 * np.pi / l) * (-(np.sqrt(3) / 2) * e1 - 0.5 * e2)
    a1, a2, a3 = a[0], a[1], a[2]
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
    pattern = genFourierPattern([1,50,0], 20)
    N = 250
    discPattern = np.real(genMesh(pattern, x = [-(N-1)/2.0, (N-1)/2.0], y = [-(N-1)/2.0, (N-1)/2.0], Nx = N, Ny = N))
    plt.imshow(discPattern,cmap='gray')
    plt.show()
