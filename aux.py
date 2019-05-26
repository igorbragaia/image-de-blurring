from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import scipy.ndimage.filters as fi
import matplotlib.cm as cm
from PIL import Image
import numpy as np


def gaussian_kernel(k_len, sigma):
    d_mat = np.zeros((k_len, k_len))
    d_mat[k_len//2, k_len//2] = 1
    return fi.gaussian_filter(d_mat, sigma)


im = Image.open('assets/pb.png')
im_grey = im.convert('L') # convert the image to *greyscale*
im_array = np.array(im_grey)

kernel = gaussian_kernel(9, 5)

plt.imshow(kernel, cmap=cm.Greys_r)
plt.savefig('assets/readme_kernel.png')


blurry_image = convolve2d(im_array, kernel)

plt.imshow(blurry_image, cmap=cm.Greys_r)
plt.savefig('assets/readme_deblurring_blurry_image.png')
