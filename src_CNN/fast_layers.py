from src_CNN.im2col import *

###########################################################################
#                             CONVULTION                                  #
###########################################################################

def conv_fast_forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    Hout = (H + 2 * pad - HH) // stride + 1
    Wout = (W + 2 * pad - WW) // stride + 1

    H += 2 * pad
    W += 2 * pad

    shape = (C, HH, WW, N, Hout, Wout)

    # im2col flatten (34x34, 34, 1, 3x34x34, 2x32, 2)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)

    # length in bytes
    strides = x.itemsize * np.array(strides)

    # DOC slide 9: creates a view into the arrfay given the exact strides and shape.
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                               shape=shape, strides=strides)

    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * Hout * Wout)

    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    res.shape = (F, N, Hout, Wout)

    # transpose to N, F, Hout, Wout
    out = res.transpose(1, 0, 2, 3)

    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def conv_backward_im2col(dout, cache):

    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)

    dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)

    return dx, dw, db


###########################################################################
#                             POOLING                                     #
###########################################################################

def max_pool_forward_fast_reshape(x, pool_param):

    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']

    x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                           W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):

    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx

def max_pool_backward_im2col(dout, cache):

    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped

    dx = im2col_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
                padding=0, stride=stride)

    dx = dx.reshape(x.shape)
    return dx