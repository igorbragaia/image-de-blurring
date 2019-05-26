from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import matplotlib.cm as cm
from PIL import Image
import numpy as np


im = Image.open('assets/pb.png')
im_grey = im.convert('L') # converter para preto e branco
im_array = np.array(im_grey)
plt.imshow(im_array, cmap=cm.Greys_r)
plt.savefig('assets/edge_before.png')


kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
result = convolve2d(im_array, kernel)
plt.imshow(result, cmap=cm.Greys_r)
plt.savefig('assets/edge_after.png')
