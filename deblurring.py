from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import scipy.optimize as optimize
import scipy.ndimage.filters as fi
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy import linalg
from scipy import sparse


def gaussian_kernel(k_len = 10, sigma = 3):
    d_mat = np.zeros((k_len, k_len))
    d_mat[k_len//2, k_len//2] = 1
    return fi.gaussian_filter(d_mat, sigma)


im = Image.open('assets/pb.png')
im_grey = im.convert('L') # convert the image to *greyscale*
im_array = np.array(im_grey)

# plt.imshow(im_array, cmap=cm.Greys_r)
# plt.savefig('assets2/deblurring_before.png')

nitems, sigma = 9, 5
kernel = gaussian_kernel(nitems, sigma)
blurry_image = convolve2d(im_array, kernel)

plt.imshow(blurry_image, cmap=cm.Greys_r)
plt.savefig('assets2/pt1_deblurring_blurry_image.png')

# Generates the toeplitz matrix for the 1D convolution
def toeplitz(b, n):
    m = len(b)
    T = np.zeros((n+m-1, n))
    for i in range(n+m-1):
        for j in range(n):
            if 0 <= i-j < m:
                T[i,j] = b[i-j]
    return T

# print(toeplitz([1,2,3], 10))

N = im_array.shape[0]

def gaussian1d(k_len = 5, sigma = 3):
    return gaussian_kernel(k_len, sigma)[k_len//2,:]

curr_1d_kernel = gaussian1d(nitems, sigma)
# Gaussian 1D kernel as matrix
T = toeplitz(curr_1d_kernel, N)

row_mat = sparse.kron(sparse.eye(N), T)
col_mat = sparse.kron(T, sparse.eye(N+nitems-1))
G = col_mat.dot(row_mat)

flat_blurry_image = blurry_image.flatten()


def lst_sq(x, A=G, b=flat_blurry_image):
    return linalg.norm(b - A.dot(x))**2


def lst_sq_grad(x, A=G, b=flat_blurry_image):
    return 2*A.T.dot(A.dot(x) - b)


# optim_output = optimize.minimize(lst_sq, np.zeros(N**2), method='L-BFGS-B', jac=lst_sq_grad, options={'disp':True})
# final_image = optim_output['x']

x = np.zeros(N ** 2)
prev_error = 0
diff = 10000
while diff > 5000:
    x -= 1e3*lst_sq_grad(x, G, flat_blurry_image)
    error = lst_sq(x, G, flat_blurry_image)
    diff = abs(error - prev_error)
    prev_error = error
    print(diff)

plt.imshow(x.reshape((N,)*2), cmap=cm.Greys_r)
plt.savefig('assets2/deblurring_deblurred_image.png')
