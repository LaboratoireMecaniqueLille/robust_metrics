#!/usr/bin/env python3

import cupy as cp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from .compute_flow import compute_flow
from . import rescale_img as ri

if __name__ == "__main__":

    xp = np

    # Cupy set_allocator
    if xp is cp:
        xp.cuda.set_allocator(
            xp.cuda.MemoryPool(xp.cuda.malloc_managed).malloc)
    # Read Images
    Im1 = cv2.imread(Path('Data/F_test4.png'), 0)
    Im2 = cv2.imread(Path('Data/F.tiff'), 0)
    print('Image shape',Im1.shape)
    #Cast
    Im1 = xp.array(Im1, dtype=xp.float32)
    Im2 = xp.array(Im2, dtype=xp.float32)
    #Initialiation
    u = xp.zeros((Im1.shape))
    v = xp.zeros((Im1.shape))
    # GNC params
    iter_gnc = 3 # Gnc iteration
    gnc_pyram_levels = 2 # Gnc levels
    gnc_factor = 1.25 # Gnc factor
    gnc_spacing = 1.25
    # Pyram params
    pyram_levels = 1
    factor = 2
    spacing = 2
    ordre_inter = 3 # Interpolation order
    # Initial weight of Energy GNC
    alpha = 1
    # Size of the median filter
    size_median_filter = 5
    # derivation Kernel for the gradient of the images
    h = xp.array([[-1, 8, 0, -8, 1]])
    h = h / 12
    # Weight for Spatial derivatives
    coef = 0.5

    # Algo params
    # Linear iterations
    max_linear_iter = 1
    max_iter = 10
    # Smoothness parameter
    lmbda = 600
    #lmbda = []
    # Metric to be used
    #metric = 'charbonnier'
    metric = 'lorentz'
    #sigma=None
    sigma = 0.05 #The parameter of Lorentz
    # Compute number of params
    pyram_levels = ri.compute_auto_pyramd_levels(Im1, spacing) #Automatic detection of the number of levels
    #pyram_levels = 5
    print('Pyram_levels', pyram_levels)
    #Mask
    Mask = None
    #Uncomment this section if you want to test the effect of the mask 
    '''Mask=np.load('Data/Masque_fissureC.npy')
    Mask=lmbda*(1-Mask)'''
    # Compute optical flow
    u, v = compute_flow(Im1, Im2, u, v, iter_gnc, gnc_pyram_levels, gnc_factor,
                        gnc_spacing, pyram_levels, factor, spacing, ordre_inter,
                        alpha, lmbda, size_median_filter, h, coef, max_linear_iter, max_iter, metric,Mask,sigma)

    #Plots
    N, M = Im1.shape
    '''y = np.linspace(0, N-1, N)
    x = np.linspace(0, M-1, M)'''
    f = xp.dstack([u,v])
    xp.save('Outputs/f_' + metric + str(lmbda), f)
    Exy, Exx = xp.gradient(u)
    Eyy, Eyx = xp.gradient(v)
    plt.figure()
    if xp is cp:
        plt.imshow(Eyy.get(), clim=(-0.05, 0.05))
    else:
        plt.imshow(Eyy, clim=(-0.05, 0.05))
    plt.title('$\lambda$=' + str(lmbda))
    plt.figure()
    if xp is cp:
       plt.imshow(Exx.get(), clim=(0, 0.006))
    else:
        plt.imshow(Exx, clim=(0, 0.006))
    plt.title('$\lambda$=' + str(lmbda))
    #plt.clim(-0.1, 0.1)
    plt.colorbar()
    plt.figure()
    if xp is cp:
        plt.imshow(f[:,:,0].get())
    else:
        plt.imshow(f[:,:,0])
    plt.colorbar()
    plt.figure()
    if xp is cp:
        plt.imshow(f[:,:,1].get())
    else:
        plt.imshow(f[:,:,1])
    plt.colorbar()
    plt.show()

