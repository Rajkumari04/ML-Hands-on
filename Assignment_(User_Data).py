#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system(' pip install scikit-learn')


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[10]:


df = pd.read_csv('User_Data.csv')
df


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df.info()


# In[14]:


df.shape


# In[15]:


sns.pairplot(df)


# In[16]:


sns.histplot(df)


# In[17]:


sns.boxplot(df)


# In[18]:


d = df.drop(['Gender'],axis = 1)
d


# In[19]:


sns.regplot(data = df,x = 'EstimatedSalary',y = 'Age')


# In[22]:


x= df.iloc[:, [2,3]].values
y= df.iloc[:, 4].values
x
y


# In[23]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)


# In[24]:


x_train


# In[26]:


x_test


# In[29]:


from sklearn.svm import SVC 
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)
classifier


# In[31]:


y_pred= classifier.predict(x_test)
y_pred


# In[33]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
cm


# In[45]:


from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step =0.01),
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[47]:


from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step =0.01),
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red','green' )))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




