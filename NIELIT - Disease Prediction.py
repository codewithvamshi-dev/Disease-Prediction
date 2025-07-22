#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[2]:


df = pd.read_csv(r'C:\Users\ravit\OneDrive\Desktop\KLH_ALL_DOX\2nd year SEM-2\INTERNSHIP\Disease Prediction\Training.csv')
df


# In[3]:


df1 = pd.read_csv(r'C:\Users\ravit\OneDrive\Desktop\KLH_ALL_DOX\2nd year SEM-2\INTERNSHIP\Disease Prediction\Testing.csv')
df1


# In[4]:


df.isnull().sum().sum()


# In[5]:


df1.isnull().sum().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


list_columns = list(df.columns)


# In[7]:


df.isnull().sum().sum()


# In[8]:


X = df.iloc[:,:-1]
X


# In[9]:


X1 = df1.iloc[:,:-1]
X1


# In[10]:


y = df.iloc[:,-1]
y


# In[11]:


'''
Y = pd.DataFrame({'Disease':Y})
    
Y'''
from sklearn import preprocessing

l =  preprocessing.LabelEncoder()
Y = l.fit_transform(y)
Y = pd.DataFrame({'Disease':Y})
Y
#print(Y.groupby(['Disease']).groups)


# In[12]:


y1 = df1.iloc[:,-1]


a1 = pd.DataFrame({'Disease': y1})


# In[13]:


'''
Y = pd.DataFrame({'Disease':Y})
    
Y'''
from sklearn import preprocessing

l =  preprocessing.LabelEncoder()
Y1 = l.fit_transform(y1)
Y1 = pd.DataFrame({'Disease':Y1})
Y1
#print(Y.groupby(['Disease']).groups)


# In[14]:


a1 = a1['Disease'].unique().tolist() #DataFrame
b1 = Y1['Disease'].unique().tolist() #Encoded Values
len(a1)


# In[15]:


def To_Dict(a1,b1):
    res_dct = {b1[i]: a1[i] for i in range(0, len(a1))}
    return res_dct



decode = To_Dict(a1,b1) 
decode


# In[16]:


X = X.values
Y = Y.values
X1 = X1.values
Y1 = Y1.values


# In[17]:


X_train,Y_train,X_test,Y_test = X,Y,X1,Y1
print(len(X_train),len(Y_train))
print(len(X_test),len(Y_test))


# In[ ]:





# In[18]:


'''from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0,random_state=0)'''


# .

# .

# # Prediction using LinearRegression

# .

# .

# In[19]:


from sklearn.linear_model import LinearRegression 
re = LinearRegression()
re.fit(X_train,Y_train)


# In[20]:


li_pred = re.predict(X_test)
li_pred = li_pred.astype(int)
li_pred


# In[21]:


from sklearn import metrics
print("mean absolute error:", metrics.mean_absolute_error(Y_test,li_pred))


# In[22]:


from sklearn.metrics import accuracy_score
li_acc = accuracy_score(li_pred,Y_test)
li_acc


# In[23]:


from sklearn.metrics import confusion_matrix
li_c = confusion_matrix(Y_test,li_pred)
li_c


# .

# .

# 
# # Prediction Using - KNN
# 

# .
# 

# .

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
m = KNeighborsClassifier(n_neighbors=3)
m.fit(X_train,Y_train)


# In[25]:


k_pred =   m.predict(X_test)
k_pred


# In[26]:


from sklearn.metrics import accuracy_score
k_acc = accuracy_score(Y_test,k_pred)
k_acc


# In[27]:


from sklearn.metrics import confusion_matrix
k_c = confusion_matrix(Y_test,k_pred)
k_c


# .

# .

# # Decision Tree Classifier

# .

# .

# In[28]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(X_train,Y_train)


# In[29]:


d_pred = dc.predict(X_test)
d_pred


# In[30]:


from sklearn.metrics import accuracy_score
d_acc = accuracy_score(Y_test,d_pred)
d_acc


# In[31]:


from sklearn.metrics import confusion_matrix
d_mat = confusion_matrix(Y_test,d_pred)
d_mat


# .

# .

# # SVM 

# .

# .

# In[32]:


from sklearn.svm import SVC
sv=SVC(kernel='linear')
sv.fit(X_train,Y_train)


# In[33]:


s_pred = sv.predict(X_test)
s_pred


# In[34]:


from sklearn.metrics import accuracy_score, confusion_matrix
s_score = accuracy_score(Y_test,s_pred)
s_score


# In[35]:


confusion_matrix(Y_test,s_pred)


# In[ ]:





# In[36]:


linear_regression_accuracy_score = li_acc*100
knn_accuracy_score = k_acc*100
DecisionTree_accuracy_score = d_acc*100
svc_accuracy_score = s_score*100

ML_Tech = ['linear_regression_accuracy_score','knn_accuracy_score','DecisionTree_accuracy_score','svc_accuracy_scor']
ML_acc = [linear_regression_accuracy_score,knn_accuracy_score,DecisionTree_accuracy_score,svc_accuracy_score]


# .

# # Graphical Analysis of different Machine Learning Algorithms

# .

# In[37]:


plt.bar(ML_Tech, ML_acc)
plt.xticks(rotation = 90)
plt.title('Accuracy Score of different machine learning techniques')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Method Type')
plt.show()


# In[38]:


plt.plot(ML_Tech, ML_acc)
plt.scatter(ML_Tech, ML_acc)
plt.xticks(rotation = 90)
plt.title('Accuracy Score of different machine learning techniques')
plt.ylabel('Accuracy Percentage')
plt.xlabel('Method Type')
plt.show()


# # Analysis of Disease and Symptoms 

# In[39]:


new_df = pd.concat([df1,df],copy=True)
#new_df.to_csv(r'C:\Users\ravit\OneDrive\Desktop\KLH_ALL_DOX\2nd year SEM-2\INTERNSHIP\Disease Prediction\merged1.csv')


# In[40]:



new_df


# .

# # Analysis of different symptoms w.r.t. disease

# .

# In[41]:


data_dis = new_df['prognosis'].unique().tolist()
#(data_dis)
data_dis.sort()
Extra = data_dis[-1]
data_dis = data_dis[:-1]


dise = np.array(data_dis).reshape(5,8)
dise
data_dis = {i:dise[i] for i in range(len(dise))
           }
data_dis = pd.DataFrame(data_dis)
data_dis['5'] = [Extra,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

print('List of Diseases:\n')
data_dis


# In[42]:


disease = new_df[new_df['prognosis']=='Diabetes']
selected = disease.groupby(['prognosis']).sum()
selected = selected.reset_index(level=['prognosis'])
selected = selected.drop(['prognosis'],axis=1)
selected = selected.T
selected = selected.rename(columns={selected.columns[0] : 'Count'})
Symptoms = list(selected[selected['Count']>0].T.columns)
print('Their are',len(Symptoms),'Different types of Symptoms in getting Chichen pox')
print('The Symptoms in getting Chicken pox are')
for i in range(len(Symptoms)):
    print(i+1,Symptoms[i])

    
    
    
p= selected[selected['Count']>0]
x_label = list(p.index)
y_label = list(p.values.flatten())
plt.plot(x_label,y_label)
plt.scatter(x_label,y_label)
plt.xticks(rotation = 90)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = True, bottom = True)
plt.title('Analisis of getting Diabetes with respect to the Symptoms')
plt.xlabel('Symptoms')
plt.show()


# In[44]:


st = '\n'.join(i for i in new_df['prognosis'].unique().tolist())
print(st,'\n\n')
s = input('Enter the name of the Disease from the given to check for Symptoms -> ')
disease = new_df[new_df['prognosis']==s]
selected = disease.groupby(['prognosis']).sum()
selected = selected.reset_index(level=['prognosis'])
selected = selected.drop(['prognosis'],axis=1)
selected = selected.T
selected = selected.rename(columns={selected.columns[0] : 'Count'})
Symptoms = list(selected[selected['Count']>0].T.columns)
print('\n\n\n\n\nTheir are',len(Symptoms),'Different types of Symptoms in getting',s)
print('The Symptoms in getting '+s+' are')
for i in range(len(Symptoms)):
    print(i+1,Symptoms[i])

    
n = input('Are you willing to plot a graph\n1.Yes\n2.No\n')
if n=='1' or n=='YES' or n=='yes' or n=='Yes':
    p= selected[selected['Count']>0]
    x_label = list(p.index)
    y_label = list(p.values.flatten())
    plt.plot(x_label,y_label)
    plt.scatter(x_label,y_label)
    plt.xticks(rotation = 90)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = True, bottom = True)
    plt.title('Analisis of getting '+s+' with respect to the Symptoms')
    plt.xlabel('Symptoms')
    plt.xticks(rotation = 90)
    plt.show()         


# In[ ]:





# In[45]:


new_df


# In[46]:


def To_Dict(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

diseases = list(new_df.columns)[:-1] 
le = len(diseases)
for i in range(1,le*2,2):
    diseases.insert(i,0)

d = To_Dict(diseases) #Dictonary of Symptoms 
d


# In[47]:


sym = list(d.keys())
sym.sort()
sym


# .

# # Analysis of different disease w.r.t. symptoms

# .

# In[48]:


ddd = np.array(sym).reshape(11,12)
ddd
data_sym = {i:ddd[i] for i in range(len(ddd))
           }
data_sym = pd.DataFrame(data_sym)
print('List of Symptoms:\n')
data_sym


# In[49]:


val = input('Enter your sub-part of the symptom: ')
print()
for i in sym:
    if val in i:
        print(i)
print('\nIf you are not satisfied with the recomendations please try to re-enter the sub-part of the symptom by re-running')

symptom = input('\n\nEnter the symptom you are facing from the  recommended list that you have got : ')


Disease = df[df[symptom] == 1].groupby(['prognosis']).sum().reset_index(level=['prognosis'])
print('\nAs you are having a symptom of '+symptom+' you might be affected the following disease')
l1 = list(Disease ['prognosis'])
for i in range(len(l1)):
    print(i+1,l1[i])


# In[ ]:





# .

# # Disease Prediction
# 

# .

# In[50]:


print('List of Symptoms:\n')
data_sym


# In[51]:



test = d


# In[52]:


test = d
d1 = {}
while True:
    n = input('Do you want to enter the symptoms that you are having:\n1. Yes\n2. No\nPlease enter your option: ')
    if n=='1' or n=='YES' or n=='yes' or n=='Yes':
        val = input('\nEnter your symptom: ')
        for i in sym:
            if val in i:
                print(i)
        s = input('Please type the symptom from above that you are facing :  ')
        d1.update({s:1})
                
    elif n=='2' or n=='no'or n=='No' or n=='NO':
        break
    else:
        continue
test.update(d1)
test  = [list(test.values())]
test = np.array(test)
print('\n\n\n')
'''#predicting using KNN
k_pred =   m.predict(test)
k_pred
'''


# In[64]:


#predicting using KNN
if len(d1)==0:
    print('To get the prediction please enter the symptoms you are having ')
k_pred =   m.predict(test)
k_pred
#prediction using SVM
s_pred = sv.predict(test)
s_pred


if k_pred == s_pred:
    print('You are having',decode[k_pred[0]])
else:
    #prediction using Linear Regression
    li_pred = re.predict(test)
    li_pred = li_pred.astype(int)
    
    #prediction using Decision Tree
    d_pred = dc.predict(test)
    
    print('The symptoms that you entered donot match any of the disease exactly but You might be having the following disease')
    print('1. ',decode[k_pred[0]],'-> Assumption using KNN Algorithm')
    print('2. ',decode[s_pred[0]],'-> Assumption using SVM Algorithm')
    print('3. ',decode[d_pred[0]],'-> Assumption using Decision Tree Algorithm')
    print('4. ',decode[li_pred[0][0]],'-> Assumption using Linear Regression Algorithm')


# In[404]:


'''
#predicting using KNN
k_pred =   m.predict(test)
k_pred'''


# In[405]:


'''
#prediction using SVM
s_pred = sv.predict(test)
s_pred
'''


# In[65]:


decode


# In[76]:


def plot_cloud(wc):
    plt.figure(figsize=(40,35))
    plt.imshow(wc)
    plt.axis('off')
    


# In[109]:


ct = ' '.join(i for i in list(decode.values()))
wc = WordCloud(background_color='white',width=2800,height=2600,random_state=1).generate(ct)

plot_cloud(wc)

plt.savefig(r'C:\Users\ravit\OneDrive\Desktop\KLH_ALL_DOX\2nd year SEM-2\INTERNSHIP\Disease Prediction\disease.png')


# In[ ]:





# In[ ]:





# In[110]:


ct1 = ' '.join(i for i in sym)
wc1 = WordCloud(background_color='white',width=2800,height=2600,random_state=1).generate(ct1)


# In[111]:


plot_cloud(wc1)
plt.savefig(r'C:\Users\ravit\OneDrive\Desktop\KLH_ALL_DOX\2nd year SEM-2\INTERNSHIP\Disease Prediction\Symptomes.png')


# In[ ]:





# In[ ]:




