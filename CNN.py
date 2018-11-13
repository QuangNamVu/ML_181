import numpy as np

from src_CNN.layers import conv_naive_forward

X = np.random.randn(100, 3, 32, 32)
# X = np.random.randint(0, 255,(100, 3, 32, 32))
conv_param = {'stride': 2, 'pad': 1}  # -> x: (N, 3, 34, 34)

w_shape = (3, 3, 4, 4)  # F =3 C = 3; HH = WW = 4

w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)

b = np.linspace(-0.1, 0.2, num=3)  # C = 3

# Hout: (34 -4)/2 + 1 -> 16 : step of sliding windows

conv_out, cache_conv = conv_naive_forward(X, w, b, conv_param)
# out: (100, 3, 16, 16)
conv_out.shape



from src_CNN.layers import max_pool_naive_forward

pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# Hout: 1+ (16 - 2)/2 => 8

pool_out, cache_pool = max_pool_naive_forward(conv_out, pool_param)
# 100, 3, 9, 9
pool_out.shape

# # Fast layer:
# + Implement fast layer in Conv and pooling
# - in file http://localhost:8888/notebooks/ML_181/ml_181/fast_layers.py

# In[4]:


from src_CNN.fast_layers import conv_fast_forward

conv_param = {'stride': 2, 'pad': 1}  # -> x: (N, 3, 34, 34)

w_shape = (3, 3, 4, 4)  # F = 3 C = 3; HH = WW = 4

out_fast, cache_fast = conv_fast_forward(X, w, b, conv_param)
out_fast.shape

# # Compare: CONV fast vs CONV naive

# In[5]:


from src_CNN.fast_layers import conv_fast_forward

from time import time

# Forward
t0 = time()
out_naive, cache_naive = conv_naive_forward(X, w, b, conv_param)
t1 = time()
out_fast, cache_fast = conv_fast_forward(X, w, b, conv_param)
t2 = time()
print('Testing conv_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))

# In[6]:


# Backward
from src_CNN.layers import conv_backward_naive
from src_CNN.fast_layers import conv_backward_im2col

t0 = time()
dx, dw, db = conv_backward_naive(out_naive, cache_naive)
t1 = time()
dx, dw, db = conv_backward_im2col(out_fast, cache_fast)
t2 = time()
print('Testing conv_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))

# # Compare: Pool fast vs Pool naive

# In[ ]:


# Pooling forward
from src_CNN.layers import max_pool_naive_forward
from src_CNN.fast_layers import max_pool_forward_fast_reshape

x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_naive_forward(x, pool_param)
t1 = time()

out_fast, cache_fast = max_pool_forward_fast_reshape(x, pool_param)
t2 = time()

print('\nTesting pool_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))

# In[ ]:


# Pooling backward

from src_CNN.layers import max_pool_backward_naive
from src_CNN.fast_layers import max_pool_backward_reshape

t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_reshape(dout, cache_fast)
t2 = time()

print('\nTesting pool_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))