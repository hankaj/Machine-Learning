#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[2]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


# In[3]:


import matplotlib.pyplot as plt
plt.imshow(X_train[142], cmap="binary")
plt.axis('off')
#plt.show()


# In[4]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[5]:


from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))


# In[6]:


model.summary()
#tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[7]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[8]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[9]:


history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_cb])


# In[10]:


import numpy as np
image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
#plt.show()


# In[11]:


#%load_ext tensorboard


# In[12]:


#%tensorboard --logdir ./image_logs


# In[13]:


model.save('fashion_clf.h5')


# In[14]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()


# In[15]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[38]:


model = keras.models.Sequential([keras.layers.Dense(30,
activation="relu",
input_shape=X_train.shape[1:]),
keras.layers.Dense(1)
])


# In[39]:


model.summary()


# In[40]:


model.compile(loss="mean_squared_error", optimizer="sgd")


# In[41]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# In[42]:


root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[43]:


history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[22]:


#%tensorboard --logdir ./housing_logs


# In[23]:


model.save("reg_housing_1.h5")


# Second model

# In[24]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(1))


# In[25]:


model.summary()


# In[26]:


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[27]:


model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[28]:


#%tensorboard --logdir ./housing_logs


# In[29]:


model.save("reg_housing_2.h5")


# Third model

# In[30]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(60, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(60, activation="relu",))
model.add(keras.layers.Dense(1))


# In[31]:


model.summary()


# In[32]:


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[33]:


model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[34]:


#%tensorboard --logdir ./housing_logs


# In[35]:


model.save("reg_housing_3.h5")

