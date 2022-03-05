#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import urllib.request


os.mkdir('data')
os.chdir('data')


# In[2]:


dir=os.path.abspath('.')  
work_path=os.path.join(dir,'housing.tgz')  
urllib.request.urlretrieve('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz', work_path)


# In[3]:


import tarfile
tf = tarfile.open('./housing.tgz')
tf.extractall()
tf.close()


# In[4]:


os.chdir('..')


# In[5]:


import gzip
with open('data/housing.csv', 'rb') as f_in, gzip.open('data/housing.csv.gz', 'wb') as f_out:
    f_out.writelines(f_in)


# In[6]:


os.remove('data/housing.tgz')


# In[7]:


os.system('gzcat data/housing.csv.gz | head -4')


# In[8]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df['ocean_proximity'].dtypes


# In[12]:


df['ocean_proximity'].value_counts()


# In[13]:


df['ocean_proximity'].describe()


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[16]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[17]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[18]:


corr = df.corr()["median_house_value"].sort_values(ascending=False)


# In[19]:


corr.reset_index().rename(columns={'index': 'atrybut', 'median_house_value': 'wspolczynnik_korelacji'}).to_csv('korelacja.csv', index=False)


# In[20]:


import seaborn as sns
sns.pairplot(df)


# In[21]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)


# In[22]:


train_set.corr()


# In[23]:


test_set.corr()


# In[24]:


# Macierze korelacji są podobne. Dane są dzielone tak, by zbiory miały podobne macierze korelacji.


# In[25]:


import pickle
with open('train_set.pkl', 'wb') as fp:
    pickle.dump(train_set, fp)
with open('test_set.pkl', 'wb') as fp:
    pickle.dump(test_set, fp)


# In[26]:


pd.options.plotting.backend = "plotly"


# In[27]:


df["median_income"].hist(figsize=(20,15))


# In[28]:


df.plot(kind="scatter", x="longitude", y="latitude")


# In[29]:


df.plot(kind="scatter", x="longitude", y="latitude",
        s=df["population"]/100, 
        c="median_house_value")

