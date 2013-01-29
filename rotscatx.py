import numpy as np
from scipy import ndimage

def rotscatx(im, p0, rot=10., sca=1., p1=([0.,0.]), order = 1 ):
    """rotate, scale and translate using one affine transformation

    rotate and scale a 2D image with optional offset

    :Parameters:
      - `im` (float) - a 2D image
      - `p0` (float) - centre point for rotation and scaling
      - `rot` (float) - rotation angle (degrees)
      - `sca` (float) - scaling
      - `p1` (float) - translation vector after rotation/scaling

    :Returns:
      - `im2` (float) - the transformed image

    :Examples:

    rotate img by 30 degrees around point 10,10 and translate by 2,2
    interpolate using splines of order 2
    >>> rotscatx(img, (-10,-10), 30., 1., ([2.,2.]), 2)

    """

    # affine discussion based on:
    # http://answerpot.com/showthread.php?3899770-%231736%3A+Proper+order+of+translation+and+rotation+in+scipy.ndimage.affine_transform

    # make a rotation matrix
    theta = np.deg2rad(rot)
    rotate = np.identity(2)

    rotate[:2,:2] = [np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]

    # make a scaling matrix
    scale = np.identity(2)
    scale *= sca

    # make affine matrix
    affine = np.dot(scale, rotate)

    # affine transform the input centre point
    p0aff = np.dot(affine, p0)

    return ndimage.interpolation.affine_transform(im, affine, offset=p0aff-p0+p1, order=order)



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    a = np.mgrid[:31,:31]
    r = np.power( (np.power(a[0]-5.,2) + np.power(a[1]-11.,2)), 2 )

    img = 10 + np.exp(- (r * r) / 2000.) * 8. + np.random.standard_normal(r.shape)

    rows,cols = img.shape

    F = plt.figure(1,(10,10))

    plt.rc('image', origin='lower', interpolation='nearest')
    plt.rc('text', usetex=True)

    gs1 = F.add_subplot(2,2,1)

    im1 = gs1.imshow(img, vmin = np.min(img), vmax=np.max(img),
        cmap=cm.Greys_r )

    gs2 = F.add_subplot(2,2,2)

    imtrans = rotscatx(img, ([-(rows/2. - 0.5), -(cols/2. - 0.5)]),
        30, 2., ([3.,0.]), 2)

    im1 = gs2.imshow(imtrans, vmin = np.min(img), vmax=np.max(img),
        cmap=cm.Greys_r )

    gs2 = F.add_subplot(2,2,3)

    imtrans = rotscatx(img, ([-(rows/2. - 0.5), -(cols/2. - 0.5)]),
        30, 0.5, ([5., 2.]), 2)

    im1 = gs2.imshow(imtrans, vmin = np.min(img), vmax=np.max(img),
        cmap=cm.Greys_r )

    gs2 = F.add_subplot(2,2,4)

    imtrans = rotscatx(img, ([-(rows/2. - 0.5), -(cols/2. - 0.5)]),
        0, 0.5, ([-10.,-10.]), 2)

    im1 = gs2.imshow(imtrans, vmin = np.min(img), vmax=np.max(img),
        cmap=cm.Greys_r )

    plt.draw()

    plt.show()
