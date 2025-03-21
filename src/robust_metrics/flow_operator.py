# coding: utf-8

import numpy as np
import cupy as cp
import cv2
try:
    from cupyx.scipy.ndimage import convolve as filter2_gpu
    from cupyx.scipy.ndimage import median_filter as median_filter_gpu
    from cupyx.scipy.sparse.linalg import LinearOperator as LinearOperator_gpu
    from cupyx.scipy.sparse.linalg import cg as cg_gpu
except Exception:
    filter2_gpu = median_filter_gpu = LinearOperator_gpu = cg_gpu = None
from scipy.ndimage import convolve as filter2_cpu
from scipy.ndimage import median_filter as median_filter_cpu
from scipy.sparse.linalg import LinearOperator as LinearOperator_cpu
from scipy.sparse.linalg import cg as cg_cpu

from . import new_fo as fo


def warp_image2(Im, xx, yy, h):
    '''
    Function to warp the second image and its derivatives
        Im: array
            The array to be warped
        xx: array
            x coordiantes
        yy: array
            y coordiantes
        h: array
            derivation kernel 
    Returns:
        WImage: array
            Warped image
        Ix: array
            Warped x derivative
        Iy: array
            Warped xy derivative
    '''
    
    xp = cp.get_array_module(Im, xx, yy, h)
    if xp is np:
        filter2 = filter2_cpu
    else:
        filter2 = filter2_gpu
    
    # We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    Im = xp.array(Im, xp.float32)
    xx = xp.array(xx, xp.float32)
    yy = xp.array(yy, xp.float32)
    if xp is cp:
        WImage = cv2.remap(Im.get(), xx.get(), yy.get(),
                           interpolation=cv2.INTER_CUBIC)
    else:
        WImage = cv2.remap(Im, xx, yy,
                           interpolation=cv2.INTER_CUBIC)

    Ix = filter2(Im, h)
    Iy = filter2(Im, h.T)

    if xp is cp:
        Iy = cv2.remap(Iy.get(), xx.get(), yy.get(), 
                       interpolation=cv2.INTER_CUBIC)
        Ix = cv2.remap(Ix.get(), xx.get(), yy.get(), 
                       interpolation=cv2.INTER_CUBIC)
    else:
        Iy = cv2.remap(Iy, xx, yy, 
                       interpolation=cv2.INTER_CUBIC)
        Ix = cv2.remap(Ix, xx, yy, 
                       interpolation=cv2.INTER_CUBIC)
    Ix = xp.array(Ix, dtype=xp.float32)
    Iy = xp.array(Iy, dtype=xp.float32)
    WImage = xp.array(WImage, dtype=xp.float32)
    return [WImage, Ix, Iy]


def derivatives(Image1, Image2, u, v, h, coef):
    '''
    Compute the derivatives 
        Image1: array
            Reference image
        Image2: array
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
    '''
    
    xp = cp.get_array_module(Image1, Image2, u, v, h)
    if xp is np:
        filter2 = filter2_cpu
    else:
        filter2 = filter2_gpu
    
    N, M = Image1.shape
    y = xp.linspace(0, N-1, N)
    x = xp.linspace(0, M-1, M)
    x, y = xp.meshgrid(x, y)
    Ix = xp.zeros((N, M))
    Iy = xp.zeros((N, M))
    x = x+u
    y = y+v
    # Derivatives of the secnd image
    WImage, I2x, I2y = warp_image2(Image2, x, y, h)
    It = WImage-Image1  # Temporal deriv
    Ix = filter2(Image1, h)  # spatial derivatives for the first image
    Iy = filter2(Image1, h.T)

    Ix = coef*I2x+(1-coef)*Ix           # Averaging
    Iy = coef*I2y+(1-coef)*Iy

    It = xp.nan_to_num(It)  # Remove Nan values on the derivatives
    Ix = xp.nan_to_num(Ix)
    Iy = xp.nan_to_num(Iy)
    out_bound = xp.where((y > N-1) | (y < 0) | (x > M-1) | (x < 0))
    Ix[out_bound] = 0  # setting derivatives value on out of bound pixels to 0
    Iy[out_bound] = 0
    It[out_bound] = 0
    return [Ix, Iy, It]


def compute_flow_base(Image1, Image2, max_iter, max_linear_iter, u, v, alpha, lmbda, size_median_filter, h, coef, mask, metric,sigma):
    '''
    Basic function to compute the displacement at each level of the pyramid
        Image1: array
            Reference image
        Image1: array
            deformed image
        max_iter: int
            number of iterations
        max_linear_iter: int
            number of linearisation iterations
        u: array
            Initial values of the horizontal displacement
        v: array
            Initial values of the vertical displacement
        alpha: float
            weight of the GNC
        lmbda: float
            Tikhonov parameter
        size_median_filter: int
            size of the median filter
        h: array
            derivation kernel 
        coef: float
            Weight for the derivatives computation
        mask: array 
            Tikhonov regularization mask
        metric: string 
            The chosen metric 'charbonnier' or 'lorentz'
        sigma: float
            parameter of the Lorentzian
    Returns: 
        u: array
            Initial values of the horizontal displacement
        v: array
            Initial values of the vertical displacement
    '''
    
    xp = cp.get_array_module(Image1, Image2, u, v, h, mask)
    if xp is np:
        median_filter = median_filter_cpu
        LinearOperator = LinearOperator_cpu
        cg = cg_cpu
    else:
        median_filter = median_filter_gpu
        LinearOperator = LinearOperator_gpu
        cg = cg_gpu
    
    N, M = u.shape
    npixels = N*M
    if metric=='lorentz' and sigma ==None:
        #print('Sigma Lorentz parameter must be defined')
        raise ValueError('Sigma Lorentz parameter must be defined')
    if lmbda<=0 :
        raise ValueError('lmbda must be >0')
    if metric!='lorentz' and metric!='charbonnier':
        raise ValueError('Undefined metric')
    for i in range(max_iter):
        print('it warping', i)
        du = np.zeros((u.shape))
        dv = np.zeros((v.shape))
        [Ix, Iy, It] = derivatives(Image1, Image2, u, v, h, coef)
        #[Ix, Iy, It] = derivatives2(Image1, Image2, u, v, h, coef)
        for j in range(max_linear_iter):
            L = LinearOperator((2*npixels, 2*npixels), matvec=lambda vector: fo.flow_matrix_final(
                u, v, du, dv, vector, N, M, Ix, Iy, It, lmbda, metric, alpha,mask,sigma))
            b = fo.flow_final_right_hand_term(Ix, Iy, It, u, v, du, dv, lmbda, metric, alpha,mask,sigma)
            #x, info = minres(L, b, tol=1e-5)
            if xp is np:
                x, info = cg(L, b, rtol=1e-1)
            else:
                x, info = cg(L, b, tol=1e-1)
            x[x > 1] = 1
            x[x < -1] = -1
            du = np.reshape(x[0:npixels], (N, M), 'F')
            dv = np.reshape(x[npixels:2*npixels],
                            (N, M), 'F')
            u0 = u
            v0 = v
            u = u+du
            v = v+dv
            if (size_median_filter != 0):
                u = median_filter(
                    u, size=(size_median_filter, size_median_filter))
                v = median_filter(
                    v, size=(size_median_filter, size_median_filter))
            du = u - u0
            dv = v - v0
            u = u0
            v = v0
        u = u + du
        v = v + dv
    return [u, v]
