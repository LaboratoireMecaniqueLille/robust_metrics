# coding: utf-8

import math
import cupy as np
from cupyx.scipy.ndimage import  gaussian_filter
from cucim.skimage.transform import resize

from . import flow_operator as fo
from . import rescale_img as ri


def compute_image_pyram_mask(Im1, ratio, N_levels):
    '''
    Function to create a pyramid of masks
    Im1: array
        The mask
    ratio: float
        The ratio
    N_levels: int
        Number of levels
    Returns:
    P1: a list
        Pyramid of mask
    '''
    P1 = []
    tmp1 = Im1
    P1.append(tmp1)

    for lev in range(1, N_levels):
        sz = np.round(np.array(tmp1.shape, dtype=np.float32)*ratio)

        tmp1 = resize(tmp1, (sz[0], sz[1]),
                    anti_aliasing=False, mode='symmetric')
        tmp1 = gaussian_filter(tmp1, 0.1)
        tmp1[tmp1<=0]=0.01
        P1.append(tmp1)
    return P1


def compute_image_pyram(Im1, Im2, ratio, N_levels, gaussian_sigma):
    '''
    Function to create a pyramid of images
    Im1: array
        The ref image
    Im2: array
        The deformed image
    ratio: float
        The ratio
    N_levels: int
        Number of levels
    gaussian_sigma: float
        Std of the gaussian filter
    Returns:
    P1: a list
        Pyramid of reference images
    P2: a list
        Pyramid of deformed images
    '''
    P1 = []
    P2 = []
    tmp1 = Im1
    tmp2 = Im2
    P1.append(tmp1)
    P2.append(tmp2)
    for lev in range(1, N_levels):
        tmp1 = gaussian_filter(tmp1, gaussian_sigma)
        tmp2 = gaussian_filter(tmp2, gaussian_sigma)
        sz = np.round(np.array(tmp1.shape, dtype=np.float32)*ratio)

        tmp1 = resize(tmp1, (sz[0], sz[1]),
                      anti_aliasing=False, mode='symmetric')
        tmp2 = resize(tmp2, (sz[0], sz[1]),
                      anti_aliasing=False, mode='symmetric')

        P1.append(tmp1)
        P2.append(tmp2)
    return [P1, P2]


def resample_flow_unequal(u, v, sz, ordre_inter):
    '''
    Function to create a reshape the displacement fields
    u: array
        The horizontal field
    v: array
        The vertical field
    sz: tuple
        The new size
    ordre_inter: int
        Order of interpolation to be used for the function resize
    Returns:
    u: array
        The reshaped horizontal field
    v: array
        The reshaped vertical field
    '''
    osz = u.shape
    ratioU = sz[0]/osz[0]
    ratioV = sz[1]/osz[1]
    u = resize(u, sz, order=ordre_inter)*ratioU
    v = resize(v, sz, order=ordre_inter)*ratioV
    return u, v


def compute_flow(Im1, Im2, u, v, iter_gnc, gnc_pyram_levels, gnc_factor,
                 gnc_spacing, pyram_levels, factor,
                 spacing, ordre_inter, alpha, lmbda, size_median_filter,
                h, coef, max_linear_iter, max_iter, metric,Mask,sigma):
    '''
    Function to compute the optical flow field
    Im1: array
        Reference image
    Im2: array
        deformed image
    u: array
        The horizontal displacement
    v: array
        The values of the vertical displacement
    h: array
        derivation kernel
    coef: float
        Weight for the derivatives computation
    Returns:
    Ix: array
        x-derivative
    Iy: array
        y-derivative
    It: array
        Temporal derivative
    alpha: float
        The weight of the quadratic formulation in the GNC process
    mask: array
        regularization mask
    sigma: float
        parameter of the Lorentzian

    '''
    P1, P2 = compute_image_pyram(
        Im1, Im2, 1/factor, pyram_levels, math.sqrt(spacing)/math.sqrt(2))
    P1_gnc, P2_gnc = compute_image_pyram(
        Im1, Im2, 1/gnc_factor, gnc_pyram_levels, math.sqrt(gnc_spacing)/math.sqrt(2))
    if isinstance(Mask,np.ndarray):
        P1_mask = compute_image_pyram_mask(Mask, 1/factor, pyram_levels)
        P1_maskgnc = compute_image_pyram_mask(Mask, 1/gnc_factor, gnc_pyram_levels)
        lmbda0=lmbda
        val=float(np.min(Mask))
    else:
        P1_mask = None
        P1_maskgnc = None
    #P1_maskgnc=compute_image_pyram_mask(Mask, 0.8, 2)
    for i in range(iter_gnc):
        print('\t \t iteration gnc', i)
        if i == 0:
            py_lev = pyram_levels
        else:
            py_lev = gnc_pyram_levels
        for lev in range(py_lev-1, -1, -1):
            if i == 0:
                Image1 = P1[lev]
                Image2 = P2[lev]
                if isinstance(Mask,np.ndarray):
                    mask_smooth = P1_mask[lev]
                sz = Image1.shape
            else:
                Image1 = P1_gnc[lev]
                Image2 = P2_gnc[lev]
                if isinstance(Mask,np.ndarray):
                    mask_smooth = P1_maskgnc[lev]
                sz = Image1.shape
            u, v = resample_flow_unequal(u, v, sz, ordre_inter)

            #keep the mask values between val and lmbda0
            if isinstance(Mask,np.ndarray):
                mask_smooth=ri.scale_image(mask_smooth,val,lmbda0)
            else:
                mask_smooth=None
            #Size median filter
            median_filter_size =size_median_filter
            #lmbda= np.hstack((mask_smooth.ravel('F'),mask_smooth.ravel('F')))
            print('IGNC',i,'alpha',alpha,'Level',lev+1, 'over', py_lev,'levels')
            u, v = fo.compute_flow_base(Image1, Image2, max_iter, max_linear_iter, u, v,
                                        alpha, lmbda, median_filter_size, h,
                                        coef, mask_smooth, metric,sigma)
            if i > 0:
                new_alpha = 1 - (i+1) / (iter_gnc)
                alpha = min(alpha, new_alpha)
                alpha = max(0, alpha)

    return u, v
