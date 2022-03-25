#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[3]:


X_cancer = data_breast_cancer["data"][["mean area", "mean smoothness"]]
X_cancer


# In[4]:


y_cancer = data_breast_cancer["target"]
y_cancer


# In[5]:


import numpy as np
data_iris = datasets.load_iris(as_frame=True)
X_iris = data_iris["data"][["petal length (cm)", "petal width (cm)"]]
y_iris = (data_iris["target"] == 2).astype(np.int8)


# In[6]:


X_iris


# In[7]:


y_iris


# In[8]:


from sklearn.model_selection import train_test_split
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)


# In[9]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
svm_cancer = LinearSVC(loss="hinge", random_state=42)
svm_cancer_scal = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge",random_state=42))])


# In[10]:


svm_cancer.fit(X_train_cancer, y_train_cancer)


# In[11]:


svm_cancer_scal.fit(X_train_cancer, y_train_cancer)


# In[12]:


y_train_pred_cancer = svm_cancer.predict(X_train_cancer)
y_test_pred_cancer = svm_cancer.predict(X_test_cancer)
y_train_pred_cancer_scal = svm_cancer_scal.predict(X_train_cancer)
y_test_pred_cancer_scal = svm_cancer_scal.predict(X_test_cancer)


# In[13]:


from sklearn.metrics import accuracy_score
train_acc_cancer = accuracy_score(y_train_cancer, y_train_pred_cancer)
test_acc_cancer = accuracy_score(y_test_cancer, y_test_pred_cancer)
train_acc_cancer_scal = accuracy_score(y_train_cancer, y_train_pred_cancer_scal)
test_acc_cancer_scal = accuracy_score(y_test_cancer, y_test_pred_cancer_scal)


# In[14]:


bc_acc = [train_acc_cancer, test_acc_cancer, train_acc_cancer_scal, test_acc_cancer_scal]


# In[15]:


bc_acc


# In[16]:


import pickle
with open('bc_acc.pkl', 'wb') as fp:
    pickle.dump(bc_acc, fp)


# In[17]:


X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)


# In[18]:


svm_iris = LinearSVC(loss="hinge", random_state=42)
svm_iris_scal = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge",random_state=42))])


# In[19]:


svm_iris.fit(X_train_iris, y_train_iris)


# In[20]:


svm_iris_scal.fit(X_train_iris, y_train_iris)


# In[21]:


y_train_pred_iris = svm_iris.predict(X_train_iris)
y_test_pred_iris = svm_iris.predict(X_test_iris)
y_train_pred_iris_scal = svm_iris_scal.predict(X_train_iris)
y_test_pred_iris_scal = svm_iris_scal.predict(X_test_iris)


# In[22]:


train_acc_iris = accuracy_score(y_train_iris, y_train_pred_iris)
test_acc_iris = accuracy_score(y_test_iris, y_test_pred_iris)
train_acc_iris_scal = accuracy_score(y_train_iris, y_train_pred_iris_scal)
test_acc_iris_scal = accuracy_score(y_test_iris, y_test_pred_iris_scal)


# In[23]:


iris_acc = [train_acc_iris, test_acc_iris, train_acc_iris_scal, test_acc_iris_scal]


# In[24]:


iris_acc


# In[25]:


with open('iris_acc.pkl', 'wb') as fp:
    pickle.dump(iris_acc, fp)

