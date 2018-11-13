import numpy as np


###########################################################################
#                             CONVULTION                                  #
###########################################################################

def conv_naive_forward(x, w, b, conv_param):
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    Hout = (W - WW + 2 * pad) // stride + 1
    Wout = (H - HH + 2 * pad) // stride + 1
    out = np.zeros((N, F, Hout, Wout))
    for idx_image, each_image in enumerate(x_padded):
        for i_H in range(Hout):
            for i_W in range(Wout):
                im_patch = each_image[:, i_H * stride:i_H * stride + HH,
                           i_W * stride:i_W * stride + WW]
                scores = (w * im_patch).sum(axis=(1, 2, 3)) + b

                out[idx_image, :, i_H, i_W] = scores

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):

    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = (W - WW + 2 * pad) // stride + 1
    Wout = (H - HH + 2 * pad) // stride + 1
    out = np.zeros((N, F, Hout, Wout))
    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    ##########################################################################
    # TODO: Implement the convolutional backward pass.                       #
    ##########################################################################
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    dx = np.zeros(x_padded.shape)

    for idx_image, image in enumerate(x_padded): # 4 sample
        for i_height in range(Hout):
            for i_width in range(Wout):
                im_patch = image[:, i_height * stride:i_height * stride + HH,
                                 i_width * stride:i_width * stride + WW]

                # duplicate to each filter F: number of filter
                im_patch = np.tile(im_patch, (F, 1, 1, 1))

                dw += (im_patch * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1))
                db += dout[idx_image, :, i_height, i_width]
                dx[idx_image:idx_image + 1, :, i_height * stride:i_height * stride + HH, i_width * stride:i_width * stride + WW] +=\
                (w * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1)).sum(axis=0)

    dx = dx[:, :, pad:-pad, pad:-pad]

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


###########################################################################
#                             POOLING                                     #
###########################################################################

def max_pool_naive_forward(x, pool_param):
    (N, C, H, W) = x.shape

    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, Hout, Wout))

    for idx_image, each_image in enumerate(x):
        for i_H in range(Hout):
            for i_W in range(Wout):
                each_window_channels = each_image[:, i_H * stride: i_H * stride + pool_height,
                                       i_W * stride: i_W * stride + pool_width]

                out[idx_image, :, i_H, i_W] = each_window_channels.max(axis=(1, 2))  # maxpooling

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################

    # Unroll variables in cache
    x, pool_param = cache

    # Get dimensions
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    # Compute dimension filters
    H_filter = (H - HH) // stride + 1
    W_filter = (W - WW) // stride + 1

    # Initialize tensor for dx
    dx = np.zeros_like(x)

    # Backpropagate dout on x
    for i in range(N):
        for z in range(C):
            for j in range(H_filter):
                for k in range(W_filter):
                    dpatch = np.zeros((HH, WW))
                    input_patch = x[i, z, j * stride:(j * stride + HH), k * stride:(k * stride + WW)]
                    idxs_max = np.where(input_patch == input_patch.max())
                    dpatch[idxs_max[0], idxs_max[1]] = dout[i, z, j, k]
                    dx[i, z, j * stride:(j * stride + HH), k * stride:(k * stride + WW)] += dpatch

    return dx
