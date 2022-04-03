#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[4]:


data=pd.read_csv('Social_Network_Ads.csv')


# In[5]:


data.head()


# In[8]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
Ad_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
Ad_classifier.fit(X_train, y_train)


# In[20]:


y_pred = Ad_classifier.predict(X_test)


# In[13]:


print(y_pred)


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test,y_pred))


# In[18]:


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[24]:


import pickle

pickle.dump(Ad_classifier, open("Ad_classifier.pkl", "wb"))

model = pickle.load(open("Ad_classifier.pkl", "rb"))


# In[25]:


print(Ad_classifier.predict(sc.transform([[30,87000]])))


# In[ ]:




