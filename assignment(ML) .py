#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[3]:


df  = pd.read_csv("heart_disease_uci.csv")


# In[4]:


df


# In[5]:


df .head()


# In[6]:


df.isnull()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['number']).columns
print("Categorical Features:\n", categorical_features)
print("Numerical Features:\n", numerical_features)


# In[11]:


df['trestbps'].fillna(df['trestbps'].mean(), inplace=True)


# In[12]:


df


# In[13]:


df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)


# In[14]:


print(df[['ca', 'thal']].isnull().sum())


# In[15]:


bins = [0, 40, 60, 120]  # Age categories: <40, 40-60, >60
labels = ['<40', '40-60', '>60']


# In[17]:


df['Age Group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)


# In[18]:


print(df[['age', 'Age Group']].head())


# In[19]:


def categorize_cholesterol(chol):
    if chol < 200:
        return 'Low'
    elif 200 <= chol <= 239:
        return 'Normal'
    else:
        return 'High'


# In[20]:


df['Cholesterol_Category'] = df['chol'].apply(categorize_cholesterol)
print(df[['chol', 'Cholesterol_Category']].head())


# In[21]:


high_cholesterol_threshold = 240  # mg/dl
high_blood_pressure_threshold = 140  # mm Hg
age_threshold = 60  # years


# In[22]:


df['high_risk'] = ((df['chol'] > high_cholesterol_threshold) | 
                     (df['trestbps'] > high_blood_pressure_threshold) | 
                     (df['age'] > age_threshold)).astype(int)


# In[23]:


print(df[['age', 'chol', 'trestbps', 'high_risk']])


# In[25]:


label_columns = ['sex', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'dataset']
label_encoders = {}
for column in label_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store the encoder for future use

print("\nData after Label Encoding:")
print(df.head())


# In[26]:


one_hot_columns = ['cp', 'dataset']  # Example columns to one-hot encode


# In[27]:


df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

print("\nData after One-Hot Encoding:")
print(df.head())


# In[28]:


from sklearn.preprocessing import LabelEncoder


# In[29]:


df = pd.read_csv('heart_disease_uci.csv')


# In[30]:


print("Original DataFrame shape:", df.shape)
print("Columns in DataFrame:", df.columns.tolist())


# In[31]:


print("Original Data:")
print(df.head())


# In[32]:


label_encoder = LabelEncoder()


# In[33]:


df['sex'] = label_encoder.fit_transform(df['sex'])


# In[35]:


if 'thal' in df.columns:
    df['thal'] = label_encoder.fit_transform(df['thal'])
else:
    print("Column 'thal' not found in DataFrame.")


if 'cp' in df.columns:
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)
else:
    print("Column 'cp' not found in DataFrame.")


bins = [0, 30, 40, 50, 60, 70, 80]
labels = ['0-30', '31-40', '41-50', '51-60', '61-70', '71-80']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)


df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)


print("\ndf after Encoding:")
print(df.head())


# In[36]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Original Data:")
print(df[['chol', 'trestbps', 'thalch']].head())


min_max_scaler = MinMaxScaler()


df[['chol', 'trestbps', 'thalch']] = min_max_scaler.fit_transform(df[['chol', 'trestbps', 'thalch']])

print("\nData after MinMax Scaling:")
print(df[['chol', 'trestbps', 'thalch']].head())


standard_scaler = StandardScaler()


df[['chol', 'trestbps', 'thalch']] = standard_scaler.fit_transform(df[['chol', 'trestbps', 'thalch']])

print("\nData after Standard Scaling:")
print(df[['chol', 'trestbps', 'thalch']].head())


print("Original Data:")
print(df[['trestbps', 'chol']].head())


df['BP_Chol_Interaction'] = df['trestbps'] * df['chol']


print("\nData after adding BP-Chol Interaction feature:")
print(df[['trestbps', 'chol', 'BP_Chol_Interaction']].head())



print("Original Data:")
print(df[['exang', 'thalch']].head())

threshold = 100


df['high_risk'] = ((df['exang'] == 'TRUE') & (df['thalch'] < threshold)).astype(int)


print("\nData after adding high_risk feature:")
print(df[['exang', 'thalch', 'high_risk']].head())


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('heart_disease_uci.csv')

print("Missing values in each column:")
print(df.isnull().sum())

df = df.dropna()

label_encoder = LabelEncoder()
categorical_columns = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop(columns=['num']) 
y = df['num']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train[['chol', 'trestbps', 'thalch']] = scaler.fit_transform(X_train[['chol', 'trestbps', 'thalch']])
X_test[['chol', 'trestbps', 'thalch']] = scaler.transform(X_test[['chol', 'trestbps', 'thalch']])


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

print(feature_importance_df)


# In[38]:


from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('heart_disease_uci.csv')

print("Original DataFrame shape:", df.shape)

label_encoder = LabelEncoder()
categorical_columns = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


var_threshold = VarianceThreshold(threshold=0.01)
var_threshold.fit(df)


features_with_variance = df.columns[var_threshold.get_support()]


df_high_variance = df[features_with_variance]

print("Shape after dropping low variance features:", df_high_variance.shape)

correlation_matrix = df_high_variance.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

threshold = 0.8
to_drop = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)


df_final = df_high_variance.drop(columns=to_drop)

print("Final DataFrame shape:", df_final.shape)
print("Dropped columns due to high correlation:", to_drop)
    


# In[ ]:




