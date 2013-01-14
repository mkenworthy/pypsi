#!/usr/bin/python

import pyfits
import numpy
from pylab import *
from scipy import *
import Image
#import MA
#import FFT
import math
import subprocess                 # For issuing commands to the OS.
import os
import sys                        # For determining the Python version.
import time
import datetime
import numpy as N
from scipy.misc import factorial as fac

def sine2d(xx,yy,amp,wavelength,a,phase):
    """
    returns value of sinewave
    """
    z = amp*sin(((cos(a)*xx+sin(a)*yy)-phase*wavelength)*2.*pi/wavelength)
    return z

def kolmomap(xx,yy,amp,wavelength,angle,phase):
    """
    stacks 2d sinewaves to reproduce a kolmogorov spectrum
    """
    sinemap=sine2d(xx,yy,amp[0],wavelength[0],angle[0]/180.*pi,phase[0])*0.
    for counter in range(len(amp)):
     sinemap=sinemap+sine2d(xx,yy,amp[counter],wavelength[counter],angle[counter]/180.*pi,phase[counter])
    return sinemap   

def app(data_pupil,data_phase,oversize=4):
"""
returns amplitude of fourier-transformed pupil amplitude and phase.
"""
    complexr=app_complex(data_pupil,data_phase,oversize)
    amp=(abs(complexr)**2)
    return amp
    
def app_complex(data_pupil,data_phase,oversize=4):
"""
returns complex image of fourier-transformed pupil amplitude and phase.
"""
#phase colors
   # cdict = {'red': ((0.0, 1.0, 1.0),(0.25, 0.0, 0.0),(0.5, 0.0, 0.0),(0.75, 1.0, 1.0),(1.00, 1.0, 1.0)),'green': ((0.0, 0.0, 0.0),(0.25, 1.0, 1.0),(0.5, 0.0, 0.0),(0.75, 1.0, 1.0),(1.0, 0.0, 0.0)),'blue': ((0.0, 0.0, 0.0),(0.25, 0.0, 0.0),(0.5, 1.0, 1.0),(0.75, 0.0, 0.0),(1.0, 0.0, 0.0))}
    #my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    size=data_pupil.shape[0]
#make empty oversized array

    expand_phase=zeros([oversize*size,oversize*size])
    expand_amp=zeros([oversize*size,oversize*size])

#copy fits into lower left corner

    expand_amp[0:size,0:size]=data_pupil[0:size,0:size]
    expand_phase[0:size,0:size]=data_phase[0:size,0:size]

#move to corners

    expand_phase=roll(expand_phase,-size/2,0)
    expand_phase=roll(expand_phase,-size/2,1)

    expand_amp=roll(expand_amp,-size/2,0)
    expand_amp=roll(expand_amp,-size/2,1)

# recalculate real and imaginary part

    #xr=expand_amp*cos(expand_phase)
    #yr=expand_amp*sin(expand_phase)

# make complex array

    complexr=expand_amp*numpy.exp(1j*expand_phase)

# apply 2d-fft

    complexr=numpy.fft.fftpack.fft2(complexr)
    return fftshift(complexr)

def circle():
  """
returns an apodized pupil of a 6.5 meter diameter telescope with 5 centimeter spacing and a central obscuration of 10%.
  """
  xmin=0
  xmax=6.5
  ymin=0.
  ymax=6.5

  x = arange(xmin, xmax, 0.005)
  y = x*1.
  [xx, yy] = meshgrid(x, y)

  zz=sqrt((xx-3.2475)**2.+(yy-3.2475)**2.)
  zz2=zz*1.
  zz2[(zz <= 3.25)]=1.
  zz2[(zz <= 0.325)]=0.
  zz2[(zz > 3.25)]=0.
  zz3=zeros(numpy.array(numpy.shape(zz2))/10)
  for i in arange(len(xx)/10):
     for j in arange(len(yy)/10):
        zz3[i,j]=numpy.sum(zz2[(i*10):(i*10+10),(j*10):(j*10+10)])/100.

  return zz3
  
def phaseangle(complexr):
"""
returns phase of complex image
"""
  return numpy.arctan2(complexr.imag,complexr.real)

def app_phase(data_pupil,data_phase,oversize=4):
"""
returns phase of fourier-transformed pupil amplitude and phase.
"""
  return phaseangle(app_complex(data_pupil,data_phase,oversize))
