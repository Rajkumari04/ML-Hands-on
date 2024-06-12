#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Dropout.csv')
df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.describe


# In[9]:


df.nunique()


# In[10]:


df.duplicated().sum()


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[14]:


df_1 = df[['Marital status','Application mode','Application order','Course','Daytime/evening attendance','Previous qualification','Nacionality',"Mother's qualification","Father's qualification","Mother's occupation","Father's occupation",'Displaced','Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder','Age at enrollment','International','Unemployment rate','Inflation rate','GDP','Target']]


# In[15]:


df_1.head(2)


# In[16]:


import warnings
warnings.filterwarnings('ignore')


# In[17]:


for i in df_1.columns:
    plt.figure(figsize=(8,4))
    plt.pie(df_1[i].value_counts(),labels=df_1[i].value_counts().index,autopct='%1.1f%%')
    hfont={'fontname':'serif','weight':'bold'}
    plt.title(i,size=20,**hfont)
    plt.show()


# In[19]:


for i in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df[i],kde=True,palette='hls')
    plt.xticks(rotation=90)
    plt.show()


# In[20]:


df_2 = df.iloc[:, :-1]


# In[21]:


for i in df_2.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(x=df['Target'],y=df_2[i],data=df,palette='hls')
    plt.show()


# In[22]:


for i in df_2.columns:
    plt.figure(figsize=(8,4))
    sns.lineplot(x=df['Target'],y=df_2[i],data=df,palette='hls')
    plt.show()


# In[23]:


for i in df_2.columns:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=df['Target'],y=df_2[i],data=df,palette='hls')
    plt.show()


# In[24]:


for i in df_2.columns:
    plt.figure(figsize=(15,6))
    pd.crosstab(index=df_2[i],columns=df['Target']).plot(kind='line')
    plt.show()


# In[26]:


df_corr = df.corr()
df_corr


# In[28]:


plt.figure(figsize=(20,17))
matrix=np.triu(df_corr)
sns.heatmap(df_corr,annot=True,linewidth=.8,mask=matrix,cmap="rocket");
plt.show()


# In[6]:


df['Target']=df['Target'].map({
'Dropout':0,
'Enrolled':1,
'Graduate':2})


# In[7]:


x=df.drop('Target',axis=1)
y=df['Target']


# In[8]:


from sklearn.ensemble import ExtraTreesClassifier


# In[9]:


model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)


# In[10]:


x=df.iloc[:,:-1]


# In[11]:


feat_importances=pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[12]:


top = pd.DataFrame({ 'Feature Importance' : feat_importances.nlargest (10)})
top


# In[13]:


x = x [['Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)','Curricular units 1st sem (grade)','Curricular units 1st sem (approved)','Tuition fees up to date','Curricular units 2nd sem (evaluations)','Curricular units 1st sem (evaluations)','Age at enrollment',"Father's occupation",'Course']]


# In[14]:


from sklearn import preprocessing


# In[15]:


scaler=preprocessing.MinMaxScaler()
x=scaler.fit_transform(x)


# In[18]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state = 42)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)


# In[21]:


ypred=lr.predict(xtest)
print("Accuracy:",accuracy_score(ytest,ypred))


# In[22]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[23]:


folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)


# In[24]:


def grid_search(model,folds,params,scoring):
    grid_search=GridSearchCV(model,cv=folds,param_grid=params,scoring=scoring,n_jobs=-1,verbose=1)

    return grid_search


# In[25]:


def print_best_score_params ( model ):
    print ( "Best Score: " , model . best_score_ )
    print ( "Best Hyperparameters: " , model . best_params_ )


# In[27]:


log_reg = LogisticRegression()
log_params = {'C':[0.01,1,10],'penalty':['l1','l2'],'solver':['liblinear','newton-cg','saga']}

grid_search_log = grid_search(log_reg,folds,log_params,scoring = None)
grid_search_log.fit(xtrain,ytrain )
print_best_score_params(grid_search_log)


# In[28]:


lr=LogisticRegression(C=10,penalty='l1',solver='saga')
lr.fit(xtrain,ytrain)


# In[29]:


ypred=lr.predict(xtest)


# In[30]:


print("Accuracy:",accuracy_score(ytest,ypred))


# In[ ]:




