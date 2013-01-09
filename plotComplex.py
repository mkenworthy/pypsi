import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import pylab as py

def plotComplex(fig, cplx_image):
    """Display amplitude and phase of a complex number

    Draws a two panel figure with the phase displayed as
    a continuous wrapping four colour bar and the amplitude
    as a grey scale plot with logrithmic intensity scale

    :Parameters:
      - `fig` (int) - a Figure container
      - `cplx_image` (cplx) - a 2D image

    :Returns:
      nothing

    :Examples:

    >>> import plotComplex
    >>> plotComplex

    Default is for 6 decades

    .. note:: Can be useful
        for important feature
    .. todo:: check that new range is not implemented

    """

    # calculate the intensity and phase
    amp = np.abs( cplx_image ) ** 2
    pha = np.angle( cplx_image )

    from matplotlib.colors import LinearSegmentedColormap

    gs = fig.add_subplot(1,2,2)

    kwargs = dict(origin="lower", interpolation="nearest")

    # red       1 0 0
    # green     0 1 0
    # blue      0 0 1
    # yellow    1 1 0
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.25, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (0.75, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.5, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (0.75, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

    phase_cmap = LinearSegmentedColormap('phase_colormap',cdict,256)

    graydict = {'red': ((0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
             'green': ( (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
             'blue': (  (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0))}

    gray_cmap = LinearSegmentedColormap('gray_colormap',graydict,256)

    im1 = gs.imshow(pha, vmin = 0, vmax = np.pi * 2, cmap=phase_cmap, **kwargs)
    # add a colourbar for phase
    cbar = plt.colorbar(im1, ticks=[0, np.pi, np.pi * 2])
    cbar.ax.set_yticklabels(['$0$','$\pi$','$2\pi$'])

    psnorm = np.log10(amp)
    psnorm -= np.max(psnorm)

    gs = fig.add_subplot(1,2,1)
    im2 = gs.imshow( psnorm, cmap=gray_cmap, vmin = -6, vmax = 0, **kwargs)
    # add a colourbar for log intensity
    cbar = plt.colorbar(im2, ticks=[0, -1, -2, -3, -4, -5, -6])


if __name__ == "__main__":

    side = 201
    rad = 15
    a = np.mgrid[:side, :side]
    b = np.mgrid[:side, :side]

    # make a simple circular aperture
    xc = (side + 1) / 2.
    yc = (side + 1) / 2.

    a[0] -= xc
    a[1] -= yc

    r = np.sqrt(a[0]*a[0] + a[1]*a[1])

    # amplitude and phase....
    amplitude = (r < rad)
    phase = np.ones_like(amplitude) * -1.

    # make it a complex number
    complx = amplitude * np.exp( 1j * phase )

    # fft it
    F1 = fftpack.fft2( fftpack.fftshift(complx) )
    F4 = fftpack.fftshift( F1 )


    # plot it all out

    F = plt.figure(1,(16,8))

    plotComplex(F, F4)

    plt.draw()
    plt.show()
