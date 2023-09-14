import cupy as np
import cv2
from cupyx.scipy.ndimage import convolve as filter2
from cupyx.scipy.ndimage import median_filter
from cupyx.scipy.sparse.linalg import LinearOperator,cg
import new_fo as fo
#import rescale_img as ri

def warp_image2(Im, xx, yy, h):
    '''
    Function to warp the second image and its derivatives
    '''
    # We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    Im = np.array(Im, np.float32)
    xx = np.array(xx, np.float32)
    yy = np.array(yy, np.float32)
    WImage = cv2.remap(Im.get(), xx.get(), yy.get(),
                       interpolation=cv2.INTER_CUBIC)

    Ix = filter2(Im, h)
    Iy = filter2(Im, h.T)

    Iy = cv2.remap(Iy.get(), xx.get(), yy.get(), interpolation=cv2.INTER_CUBIC)
    Ix = cv2.remap(Ix.get(), xx.get(), yy.get(), interpolation=cv2.INTER_CUBIC)
    Ix = np.array(Ix, dtype=np.float32)
    Iy = np.array(Iy, dtype=np.float32)
    WImage = np.array(WImage, dtype=np.float32)
    return [WImage, Ix, Iy]

def derivatives(Image1, Image2, u, v, h, b):
    '''
    Compute the derivatives 
    '''
    N, M = Image1.shape
    y = np.linspace(0, N-1, N)
    x = np.linspace(0, M-1, M)
    x, y = np.meshgrid(x, y)
    Ix = np.zeros((N, M))
    Iy = np.zeros((N, M))
    x = x+u
    y = y+v
    # Derivatives of the secnd image
    WImage, I2x, I2y = warp_image2(Image2, x, y, h)
    It = WImage-Image1  # Temporal deriv
    Ix = filter2(Image1, h)  # spatial derivatives for the first image
    Iy = filter2(Image1, h.T)

    Ix = b*I2x+(1-b)*Ix           # Averaging
    Iy = b*I2y+(1-b)*Iy

    It = np.nan_to_num(It)  # Remove Nan values on the derivatives
    Ix = np.nan_to_num(Ix)
    Iy = np.nan_to_num(Iy)
    out_bound = np.where((y > N-1) | (y < 0) | (x > M-1) | (x < 0))
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
    N, M = u.shape
    npixels = N*M
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
