#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)


# In[2]:


import pandas as pd
pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[3]:


from sklearn.model_selection import train_test_split
X = iris.data[["petal length (cm)", "petal width (cm)"]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[4]:


from sklearn.linear_model import Perceptron
per_clf_0 = Perceptron()
y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)

per_clf_1 = Perceptron()
y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)

per_clf_2 = Perceptron()
y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)


# In[5]:


per_clf_0.fit(X_train, y_train_0)
y_pred_train_0 = per_clf_0.predict(X_train)
y_pred_test_0 = per_clf_0.predict(X_test)

per_clf_1.fit(X_train, y_train_1)
y_pred_train_1 = per_clf_1.predict(X_train)
y_pred_test_1 = per_clf_1.predict(X_test)

per_clf_2.fit(X_train, y_train_2)
y_pred_train_2 = per_clf_2.predict(X_train)
y_pred_test_2 = per_clf_2.predict(X_test)


# In[6]:


from sklearn.metrics import accuracy_score
train_acc_0 = accuracy_score(y_train_0, y_pred_train_0)
test_acc_0 = accuracy_score(y_test_0, y_pred_test_0)

train_acc_1 = accuracy_score(y_train_1, y_pred_train_1)
test_acc_1 = accuracy_score(y_test_1, y_pred_test_1)

train_acc_2 = accuracy_score(y_train_2, y_pred_train_2)
test_acc_2 = accuracy_score(y_test_2, y_pred_test_2)


# In[7]:


per_acc = [(train_acc_0, test_acc_0), (train_acc_1, test_acc_1), (train_acc_2, test_acc_2)]
per_acc


# In[8]:


per_wght = []
for perceptron in [per_clf_0, per_clf_1, per_clf_2]:
    w_0 = perceptron.intercept_[0]
    w_1 = perceptron.coef_[0, 0]
    w_2 = perceptron.coef_[0, 1]
    per_wght.append((w_0, w_1, w_2))


# In[9]:


per_wght


# In[10]:


import pickle
with open('per_acc.pkl', 'wb') as fp:
    pickle.dump(per_acc, fp)
with open('per_wght.pkl', 'wb') as fp:
    pickle.dump(per_wght, fp)


# In[11]:


import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])


# In[12]:


import keras
import tensorflow as tf
model = keras.models.Sequential()
model.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))


# In[13]:


model.summary()


# In[14]:


model.compile(loss="binary_crossentropy",
optimizer="sgd")


# In[15]:


history = model.fit(X, y, epochs=100, verbose=False)
print(history.history['loss'])


# In[16]:


model.predict(X)


# In[17]:


model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model2.add(keras.layers.Dense(1, activation="sigmoid"))
model2.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
history = model2.fit(X, y, epochs=100, verbose=False)


# In[18]:


model2.predict(X)


# In[19]:


model3 = keras.models.Sequential()
model3.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model3.add(keras.layers.Dense(1, activation="sigmoid"))
model3.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), metrics=['binary_accuracy'])
history = model3.fit(X, y, epochs=100, verbose=False)


# In[20]:


model3.predict(X)


# In[21]:


found = False
while not found:
    model4 = keras.models.Sequential()
    model4.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
    model4.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))
    model4.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=["accuracy"])
    history = model4.fit(X, y, epochs=100, verbose=False)
    results = model4.predict(X)
    if results[0]<0.1 and results[1]>0.9 and results[2]>0.9 and results[0]<0.1:
        found = True


# In[22]:


results


# In[23]:


weights=model4.get_weights()


# In[24]:


with open('mlp_xor_weights.pkl', 'wb') as fp:
    pickle.dump(weights, fp)

