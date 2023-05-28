#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


test_data=pd.read_csv("C:/Users/VAISHNAV/Downloads/test_lAUu6dG.csv")
train_data=pd.read_csv("C:/Users/VAISHNAV/Downloads/train_ctrUa4K.csv")


# In[3]:


test_data


# In[4]:


train_data


# In[5]:


d=train_data.drop('Loan_ID',axis=1)
k=test_data.drop('Loan_ID',axis=1)


# In[6]:


d


# In[7]:


d.isnull().sum()


# In[8]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
d['Gender']=labelencoder.fit_transform(d['Gender'])
d['Education']=labelencoder.fit_transform(d['Education'])
d['Self_Employed']=labelencoder.fit_transform(d['Self_Employed'])
d['Married']=labelencoder.fit_transform(d['Married'])
d['Property_Area']=labelencoder.fit_transform(d['Property_Area'])
d['Loan_Status']=labelencoder.fit_transform(d['Loan_Status'])
k['Gender']=labelencoder.fit_transform(k['Gender'])
k['Education']=labelencoder.fit_transform(k['Education'])
k['Self_Employed']=labelencoder.fit_transform(k['Self_Employed'])
k['Married']=labelencoder.fit_transform(k['Married'])
k['Property_Area']=labelencoder.fit_transform(k['Property_Area'])



# In[9]:


d


# In[10]:


k


# In[11]:


d['Dependents']=d['Dependents'].replace(to_replace="3+",value='4')
d


# In[12]:


d.isnull().sum()


# In[13]:


for i in ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']:
    d[i]=d[i].fillna(d[i].median())


# In[14]:


d


# In[15]:


k.isnull().sum()


# In[16]:


k['Dependents']=k['Dependents'].replace(to_replace="3+",value='4')
k


# In[17]:


for i in ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']:
    k[i]=k[i].fillna(k[i].median())


# In[18]:


k.isnull().sum()


# In[19]:


d.isnull().sum()


# In[20]:


d.shape


# In[21]:


k.shape


# In[22]:


X1=d.iloc[0:367,11].values


# In[23]:


X1


# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

                            

                            


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(k,X1,test_size=0.2, random_state=42)


model = RandomForestRegressor()


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


y_pred = best_model.predict(k)
y_pred


# In[ ]:





# In[109]:


import numpy as np
values = np.round(y_pred)
values


# In[110]:


sample=pd.read_csv("C:/Users/VAISHNAV/Downloads/sample_submission_49d68Cx.csv")
result=pd.DataFrame(values,columns=['Loan_Status'])
result


# In[111]:


type(result['Loan_Status'])


# In[112]:


sample


# In[113]:


sample['Loan_Status'] = result['Loan_Status']
sample


# In[114]:


sample['Loan_Status'] = sample['Loan_Status'].replace(1, 'Y')
sample['Loan_Status'] = sample['Loan_Status'].replace(0, 'N')
sample


# In[115]:


fnl_csv_data = sample.to_csv('answer7.csv', index = True)


# In[ ]:





# In[ ]:





# In[ ]:




