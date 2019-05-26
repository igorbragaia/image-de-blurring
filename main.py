import pylab as pl
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from pprint import pprint
from scipy.stats import norm
from scipy.linalg import toeplitz
from scipy import fftpack

def gkern(kernlen=5, nsig=3):
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(norm.cdf(x))
    return kern1d

def gkern_toeplitz(kernlen=5, nsig=3):
    return toeplitz(gkern(kernlen, nsig))

def kron(a, b):
    return np.kron(np.eye(2), np.ones((2,2)))

def gfilter(n,m):
    T = gkern_toeplitz(n,3)
    return np.dot(kron(T,np.identity(m+n)),kron(np.identity(m),T))

class DeBlurringImage:
    def __init__(self, path):
        self.im_array = None
        self.load_image(path)

    def load_image(self,path):
        im = Image.open(path)
        im_grey = im.convert('L') # convert the image to *greyscale*
        im_array = np.array(im_grey)
        self.im_array = im_array

    def display_image(self):
        pl.imshow(self.im_array, cmap=cm.Greys_r)
        pl.show()

    def deblur(self):
        # # n = self.im_array.shape[0]
        # # m = 5
        # # G = gfilter(n,m)
        pass

# deblurring = DeBlurringImage('blurred.png')
# deblurring.deblur()
# deblurring.display_image()

print(gkern_toeplitz())
