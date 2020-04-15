#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8


# In[20]:


i=20 # este es uno de esos digitos
_ = plt.imshow(imagenes[i])
plt.title('{}'.format(target[i]))
print(imagenes[i])


# In[21]:


# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))


# In[16]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[41]:


# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


# In[42]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[43]:


# Turn up tolerance for faster convergence
train_samples = int(n_imagenes/2)
#regresión logística sobre los dígitos
clf = LogisticRegression(
    C=50. / train_samples, penalty='l1', solver='saga', tol=0.1)#,multi_class='multinomial'
clf.fit(x_train, y_train)
#predicciones sobre los valores de los dígitos
y_pred=clf.predict(x_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(x_test, y_test)


# In[44]:


# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)


# In[45]:


coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')
plt.savefig("coeficientes.png")


# In[46]:


np.shape(coef)


# In[49]:


#confusion_matrix(y_true, y_pred)
plt.imshow(confusion_matrix(y_train, y_pred))
plt.savefig("confusion.png")


# In[36]:


y_train


# In[38]:


np.shape()


# In[39]:


np.shape(x_train)


# In[ ]:




