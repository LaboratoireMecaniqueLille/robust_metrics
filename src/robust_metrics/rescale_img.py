# coding: utf-8

import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate


def scale_image(Image,vlow,vhigh):
    '''
    Keep the values of Image in[vlow,vhigh]
    Params:
        Image: array
            The image
        vlow: float
            lower value
        vhigh: float
            higher value
    Return:
        imo: array
            image with values between vlow and vhigh
    '''
    ilow= np.min(Image)
    ihigh= np.max(Image)
    if ilow==ihigh:
        imo=Image
    imo = (Image-ilow)/(ihigh-ilow) * (vhigh-vlow) + vlow
    return  imo


def decompo_texture(im, theta, nIters, alp, isScale):
    '''
    Perform the decomposition texture structure
    for the definition of the parameters please refer
    to the the paper of Wedel
    Params:
        im: array
            The image
        theta: float
            parameter of the algorithm
        nIters: int
            Number of iterations
        vhigh: float
            higher value
        alp: float
            parameter of the algorithm
        isScale: bool
            to rescale the image using scale_image
    Return:
        texture: array
            the texture part of im
        structure: array
            the structure part of im
    '''
    IM   = scale_image(im, -1,1)
    im= np.copy(IM)

    p=np.zeros((im.shape[0],im.shape[1],2),dtype=np.float32)
    delta = 1.0/(4.0*theta)
    I=np.squeeze(IM)
    for iter in range (nIters):

        #Compute divergence        eqn(8)
        div_p =correlate(p[:,:,0], np.array([[-1, 1, 0]]),mode='wrap' )+ correlate(p[:,:,1], np.array( [[-1]  , [1], [0]]), mode='wrap')

        I_x = filter2(I+theta*div_p, np.array([[1, -1]]))

        I_y = filter2(I+theta*div_p, np.array([ [1],[-1] ]))

        # Update dual variable      eqn(9)
        p[:,:,0] = p[:,:,0] + delta*I_x
        p[:,:,1] = p[:,:,1] + delta*I_y

        # Reproject to |p| <= 1     eqn(10)

        reprojection = np.maximum(1.0,  np.sqrt(  p[:,:,0]**2+ p[:,:,1]**2 ))

        #print('repre',reprojection)
        p[:,:,0] = p[:,:,0]/reprojection
        p[:,:,1] = p[:,:,1]/reprojection
        #print(p[:im.shape[0],:im.shape[1],0])

    # compute divergence
    div_p = correlate(p[:,:,0], np.array([[-1, 1, 0]] ),mode='wrap' ) +  correlate(p[:,:,1], np.array( [[-1]  , [1], [0]]),mode='wrap')

    #compute structure component
    IM[:,:] = I + theta*div_p
    if (isScale):
        texture   = np.squeeze(scale_image((im - alp*IM), 0, 255))

        structure = np.squeeze(scale_image(IM, 0, 255))
    else:
        texture   = np.squeeze(im - alp*IM)
        structure = np.squeeze(IM)

    return [texture, structure]


def compute_auto_pyramd_levels(Im,spacing):
    '''
    Determine dynamically the number of parameters 
    using the Im shape
    Params:
        Im: array
            the image
        spacing: float
            the downsampling factor of the pyramid
    Returns: 
        pyramid_levels: int
            number of levels
    '''
    N1 = 1 + math.floor( math.log(max(Im.shape[0], Im.shape[1])/16)/math.log(spacing) )
    #smaller size shouldn't be less than 6
    N2 = 1 + math.floor( math.log(min(Im.shape[0], Im.shape[1])/6)/math.log(spacing) )
    pyramid_levels  =  min(N1, N2)
    '''if this.old_auto_level
        this.pyramid_levels  =  1 + floor( log(min(size(images, 1),...
            size(images,2))/16) / log(this.pyramid_spacing) );'''
    return pyramid_levels


if __name__=="__main__":
    #Image= cv2.imread('24-3_45degReslice/24-3_110_45degReslice.tif', 0)
    Image= cv2.imread('24-3_45degReslice/24-3_110_45degReslice_translated.tif', 0)
    Image=Image.astype(np.float32)
    theta   = 1/8
    nIters  = 100
    alp= 0.95
    isScale=True
    [tex,stru]=decompo_texture(Image, theta, nIters, alp, isScale)
