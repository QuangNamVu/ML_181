#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)




# # Convert data numeric -> category

# In[5]:


def split(arr, cond):
    return (arr[cond], arr[~cond])


# In[8]:


num_att = X.shape[1]
pivot = np.mean(X, axis=0)
pivot

# In[21]:


X_category = np.full(X.shape, True)

for iteration in range(num_att):
    X_category[:, iteration] = X[:, iteration] > pivot[iteration]

X_category

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# # Define Infomation Gain

# In[25]:


def entropy(l_summary):
    # normalize
    l_summary = l_summary / l_summary.sum(axis=0, keepdims=1)
    l_summary += np.finfo(np.float32).eps

    return (-np.sum(l_summary * np.log2(l_summary)))


# In[26]:


class Tree(object):
    def __init__(self):
        self.data = None
        self.best_attribute = -1
        self.child = []
