'''
Filename: time_lapse.py
Author: Matthew Peres Tino
Description: Uses functions from pattern_generation.py to generate random patterns in a time lapse fashion.

'''
# packages
import time
import random 
import numpy as np
import matplotlib.pyplot as plt
import pattern_generation as pgen

# variables to consider
imageSize = 250 # length and width of pattern images
sizeDisc = 60 # maximum wavelength of pattern for scale randomization

# first we need to randomize our variables using numpy


# now time lapse of plots
for i in range(100):
    m = random.randrange(2, 8, 2) # generate random from 2, 4, 6. It is not end inclusive
    theta = random.random() * (2*np.pi/m)
    scale = (random.random() * sizeDisc) + 5 # ensure minimum 5

    # now call genFourierPattern
    pattern = pgen.genFourierPattern(sym=m, lamb=scale, theta=theta)
    discPattern = np.real(pgen.genMesh(pattern, x = [-(imageSize-1)/2.0, \
                            (imageSize-1)/2.0], y = [-(imageSize-1)/2.0, (imageSize-1)/2.0], Nx = imageSize, Ny = imageSize))
    
    # time lapse
    plt.imshow(discPattern,cmap='gray')
    plt.axis('off')
    plt.title('{}-fold symmetry, angle = {} rads'.format(m, np.round(theta,2)))
    plt.show(block=False)
    plt.pause(0.75)
    plt.close()