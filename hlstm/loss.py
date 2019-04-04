import numpy as np
import torch
import torch.nn.functional as fnn
from torch import nn
import torch.optim as optim
from settings import DEVICE as device
# from util import maybe_cuda,maybe_parallel

# is n_channels in_channels or out_channels?
# or we want out_channels == in_channels?

############ LAPLACIAN PYRAMID LOSS #########################

def build_gauss_kernel(size=(5, 3), sigma=1.0, n_channels=1):
    if (type(size) == int):
        height = width = size
    else:
        height, width = size[0], size[1]
    if (type(sigma) == float):
        h_sigma = w_sigma = sigma
    else:
        h_sigma, w_sigma = sigma[0], sigma[1]

    if (height % 2 != 1 or width % 2 != 1):
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:width, 0:height].T)
    '''
    grid: 
    [(0, 0), (0, 1), (0, 2),
     (1, 0), (1, 1), ......,
     (2, 0), (2, 1), ......]
    '''

    def gaussian(x, size, stddev): return np.exp((x - size // 2)**2 / (-2 * stddev**2))**2
    h_kernel = gaussian(grid[:, :, 1], height, h_sigma)
    w_kernel = gaussian(grid[:, :, 0], width, w_sigma)
    kernel = h_kernel + w_kernel
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1)) # [n_channels, size, size]
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :]).to(device)
    kernel.requires_grad_(False)

    return kernel

def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kh // 2, kh // 2, kw // 2, kw // 2), mode='replicate')
    # Pad width dim by (kw // 2, kh // 2) and height dim by (kw // 2, kh // 2)
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for _ in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2) # subsampling

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):

    def __init__(self, max_levels=3, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.lossfn=nn.MSELoss() # fnn.l1_loss
        self.k_size=k_size
        self.sigma=sigma
        self.max_levels=max_levels
        self._gauss_kernel=None

    def forward(self, input, target):
        # input: [N, C, H, W]
        input = input[:, None, :, :]
        target = target[:, None, :, :]
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1]
            )
        pyr_input = laplacian_pyramid(
            input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(
            target, self._gauss_kernel, self.max_levels)
        result = sum(self.lossfn(a, b) for a, b in zip(pyr_input, pyr_target))
        result /= self.max_levels + 1
        return result

############ LAPLACIAN PYRAMID LOSS #########################

############ MSE LOSS #######################################

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.lossfn=nn.MSELoss(reduce=False).to(device)

    def forward(self, input, target):
        mse_matrix = self.lossfn(input, target)
        return result

############ MSE LOSS #######################################

if __name__ == "__main__":
    lapLoss = LapLoss(2, (9, 3), (1.5, 0.5)).to(device)
    N, H, W = 1, 10, 10

    input = torch.randn(N, H, W).to(device)
    input.requires_grad_(True)
    optimizer = optim.SGD([input], lr=0.01)

    target = torch.randn(N, H, W).to(device)
    loss = lapLoss.forward(input, target)
    loss.backward()
    optimizer.step()

    # [128, 256] [100, 200, 400]

    # [128, 100] -> [32, 25]
    # [128, 200] -> [16, 25]
    # [128, 400] -> [8,  25]
