#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
from sklearn.datasets import load_iris
data_iris = load_iris(as_frame=True)


# In[2]:


X_bc = data_breast_cancer["data"]
X_bc


# In[3]:


X_ir = data_iris["data"]
X_ir


# In[4]:


from sklearn.decomposition import PCA
pca_bc = PCA(n_components=0.9)
pca_ir = PCA(n_components=0.9)


# In[5]:


X_bc_pca = pca_bc.fit_transform(X_bc)
X_ir_pca = pca_ir.fit_transform(X_ir)


# In[6]:


print(X_bc.shape, '->', X_bc_pca.shape)
print(X_ir.shape, '->', X_ir_pca.shape)


# In[7]:


pca_bc.explained_variance_ratio_


# In[8]:


pca_ir.explained_variance_ratio_


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler_bc = StandardScaler()
scaler_ir = StandardScaler()


# In[10]:


X_bc_scal = scaler_bc.fit_transform(X_bc)
X_ir_scal = scaler_ir.fit_transform(X_ir)


# In[11]:


pca_bc_scal = PCA(n_components=0.9)
pca_ir_scal = PCA(n_components=0.9)
X_bc_pca_scal = pca_bc_scal.fit_transform(X_bc_scal)
X_ir_pca_scal = pca_ir_scal.fit_transform(X_ir_scal)


# In[12]:


print(X_bc.shape, '->', X_bc_pca_scal.shape)
print(X_ir.shape, '->', X_ir_pca_scal.shape)


# In[13]:


variance_ratio_bc = list(pca_bc_scal.explained_variance_ratio_)
variance_ratio_bc


# In[14]:


variance_ratio_ir = list(pca_ir_scal.explained_variance_ratio_)
variance_ratio_ir


# In[15]:


import pickle
with open('pca_bc.pkl', 'wb') as fp:
    pickle.dump(variance_ratio_bc, fp)
with open('pca_ir.pkl', 'wb') as fp:
    pickle.dump(variance_ratio_ir, fp)


# In[16]:


import numpy as np
idx_bc = [np.abs(component).argmax() for component in pca_bc_scal.components_]
idx_ir = [np.abs(component).argmax() for component in pca_ir_scal.components_]


# In[17]:


idx_bc


# In[18]:


idx_ir


# In[19]:


with open('idx_bc.pkl', 'wb') as fp:
    pickle.dump(idx_bc, fp)
with open('idx_ir.pkl', 'wb') as fp:
    pickle.dump(idx_ir, fp)

