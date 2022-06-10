#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[2]:


X_train.shape


# In[3]:


from tensorflow import keras
def build_model(n_hidden, n_neurons, optimizer, learning_rate, momentum=0):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    if optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    if optimizer == "momentum":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", metrics=["mae"], optimizer=optimizer)
    return model


# In[4]:


import os
import time
root_logdir = os.path.join(os.curdir, "tb_logs")

def get_dir(name, value):
    ts = int(time.time())
    run_id = f"{ts}_{name}_{value}"
    return os.path.join(root_logdir, run_id)


# In[5]:


es_cb = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.00)


# In[6]:


import numpy as np
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# First experiment

# In[7]:


lr = [10**(-6), 10**(-5), 10**(-4)]
name = "lr"
results_lr = []
for value in lr:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_dir(name, value))
    model = build_model(1, 25, "sgd", value)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es_cb])
    score = model.evaluate(X_test, y_test)
    results_lr.append((value, score[0], score[1]))


# In[8]:


results_lr


# In[9]:


import pickle
with open('lr.pkl', 'wb') as fp:
    pickle.dump(results_lr, fp)


# In[10]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('tensorboard', '--logdir ./tb_logs')


# In[11]:


hl = list(range(4))
name = "hl"
results_hl = []
for value in hl:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_dir(name, value))
    model = build_model(value, 25, "sgd", 10**(-5))
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es_cb])
    score = model.evaluate(X_test, y_test)
    results_hl.append((value, score[0], score[1]))


# In[12]:


with open('hl.pkl', 'wb') as fp:
    pickle.dump(results_hl, fp)
    
results_hl


# In[13]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./tb_logs')


# In[14]:


nn = [5, 25, 125]
name = "nn"
results_nn = []
for value in nn:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_dir(name, value))
    model = build_model(1, value, "sgd", 10**(-5))
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es_cb])
    score = model.evaluate(X_test, y_test)
    results_nn.append((value, score[0], score[1]))


# In[15]:


with open('nn.pkl', 'wb') as fp:
    pickle.dump(results_nn, fp)
    
results_nn


# In[16]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./tb_logs')


# In[17]:


opt = ["sgd", "nesterov", "momentum", "adam"]
name = "opt"
results_opt= []
for value in opt:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_dir(name, value))
    model = build_model(1, 25, value, 10**(-5), 0.5)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es_cb])
    score = model.evaluate(X_test, y_test)
    results_opt.append((value, score[0], score[1]))


# In[18]:


with open('opt.pkl', 'wb') as fp:
    pickle.dump(results_opt, fp)
    
results_opt


# In[19]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./tb_logs')


# In[20]:


mom = [0.1, 0.5, 0.9]
name = "mom"
results_mom = []
for value in mom:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_dir(name, value))
    model = build_model(1, 25, "momentum", 10**(-5), value)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tensorboard_cb, es_cb])
    score = model.evaluate(X_test, y_test)
    results_mom.append((value, score[0], score[1]))


# In[21]:


with open('mom.pkl', 'wb') as fp:
    pickle.dump(results_mom, fp)
    
results_mom


# In[22]:


#get_ipython().run_line_magic('tensorboard', '--logdir ./tb_logs')


# In[23]:


param_distribs = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": [5, 25, 125],
    "model__learning_rate": [10**-6, 10**-5, 10**-4],
    "model__optimizer": ["sgd", "nesterov", "momentum", "adam"],
    "model__momentum": [0.1, 0.5, 0.9]
}


# In[24]:


import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)

keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[25]:


from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=20, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)


# In[26]:


rnd_search_cv.best_params_


# In[27]:


with open('rnd_search.pkl', 'wb') as fp:
    pickle.dump(rnd_search_cv.best_params_, fp)

