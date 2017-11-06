
# coding: utf-8

# Q1:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
mean2 = [3, 0]
cov2 = [[1, 0], [0, 1]]
mean3 = [0, 3]
cov3 = [[1, 0], [0, 1]]

# generate data

n = 500
data = np.zeros(shape=(2,n))
for i in range(n):
    a = np.random.uniform(0,1)
    if a <= .2:
        x = np.random.multivariate_normal(mean1, cov1, 1)
    elif .2 < a <= .7:
        x = np.random.multivariate_normal(mean2, cov2, 1)
    else:
        x = np.random.multivariate_normal(mean3, cov3, 1)
        
    data[:,i] = x
    
data = pd.DataFrame(data)
data


# In[97]:

plt.plot(data.iloc[0], data.iloc[1], 'o')
plt.axis('equal')
plt.show()


# In[98]:

def kmean(K, iterate, mu, c, L):
    for l in range(iterate):
        # update c
        for j in range(len(c)):
            mini = 9999999999
            for k in range(K):
                sqr_diff = sum((data[j].values - mu[k].values)**2)
                if sqr_diff < mini:
                    c[j] = k
                    mini = sqr_diff
        # update mu
        for k in range(K):
            n_k = c.count(k)
            mu[k] = data[[i for i,x in enumerate(c) if x == k]].sum(axis=1).values * 1/n_k 

        # objective function  
        obj = 0
        for x in range(len(c)):
            for k in range(K):
                if c[x] == k:
                    obj = obj+sum((data[x].values - mu[k].values)**2)
        L[l] = obj
    return L


# In[99]:

K2 = 2
iterate = 20
mu2 = [[1,0],[1,0]]
mu2 = pd.DataFrame(mu2)
c2 = [None] * n
L2 = [None] * iterate
L2 = kmean(K2, iterate, mu2, c2, L2)

K3 = 3
mu3 = [[1,3,2], [1,2,3]]
mu3 = pd.DataFrame(mu3)
c3 = [None] * n
L3 = [None] * iterate
L3 = kmean(K3, iterate, mu3, c3, L3)

K4 = 4
mu4 = [[1,4,3,2], [1,2,3,4]]
mu4 = pd.DataFrame(mu4)
c4 = [None] * n
L4 = [None] * iterate
L4 = kmean(K4, iterate, mu4, c4, L4)

K5 = 5
mu5 = np.array([[1,2],[0,2],[3,3],[1,4],[5,0]])
mu5 = pd.DataFrame(mu5.T)
c5 = [None] * n
L5 = [None] * iterate
L5 = kmean(K5, iterate, mu5, c5, L5)


# In[100]:

x = np.arange(1,21,1)
K2 = plt.scatter(x=x,y= L2)
K3 = plt.scatter(x=x,y= L3)
K4 = plt.scatter(x=x,y= L4)
K5 = plt.scatter(x=x,y= L5)
plt.legend((K2, K3, K4, K5),
           ('K=2', 'K=3', 'K=4', 'K=5'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=10)
plt.xlabel("Iterations",fontsize=12)
plt.ylabel("K-means Objective Function",fontsize=12)
plt.xlim(-1,22)
plt.show()


# In[101]:

def cla(K, iterate, mu, c, L):
    for l in range(iterate):
        # update c
        for j in range(len(c)):
            mini = 9999999999
            for k in range(K):
                sqr_diff = sum((data[j].values - mu[k].values)**2)
                if sqr_diff < mini:
                    c[j] = k
                    mini = sqr_diff
        # update mu
        for k in range(K):
            n_k = c.count(k)
            mu[k] = data[[i for i,x in enumerate(c) if x == k]].sum(axis=1).values * 1/n_k 

        # objective function  
        obj = 0
        for x in range(len(c)):
            for k in range(K):
                if c[x] == k:
                    obj = obj+sum((data[x].values - mu[k].values)**2)
        L[l] = obj
    return c

K3 = 3
mu3 = [[1,3,2], [1,2,3]]
mu3 = pd.DataFrame(mu3)
c3 = [None] * n
L3 = [None] * iterate
c3 = cla(K3, iterate, mu3, c3, L3)
K5 = 5
mu5 = np.array([[1,2],[0,2],[3,3],[1,4],[5,0]])
mu5 = pd.DataFrame(mu5.T)
c5 = [None] * n
L5 = [None] * iterate
c5 = cla(K5, iterate, mu5, c5, L5)


# In[102]:

plt.scatter(data.iloc[0], data.iloc[1], c=c3)
plt.
(mu3.iloc[0],mu3.iloc[1], 'r>')
plt.axis('equal')
plt.xlabel("x1",fontsize=12)
plt.ylabel("x2",fontsize=12)
plt.show()


# In[103]:

plt.scatter(data.iloc[0], data.iloc[1], c=c5)
plt.plot(mu5.iloc[0],mu5.iloc[1], 'r>')
plt.axis('equal')
plt.xlabel("x1",fontsize=12)
plt.ylabel("x2",fontsize=12)
plt.show()


# In[ ]:




# In[2]:

import pandas as pd
import numpy as np
from numpy.linalg import inv

column_names = ['user_id', 'movie_id', 'rating']
data =  pd.read_csv("/Users/sihui/Desktop/COMS4721ML/HW/HW4/COMS4721_hw4-data/ratings.csv",
                   header=None, names = column_names)
data_test = pd.read_csv("/Users/sihui/Desktop/COMS4721ML/HW/HW4/COMS4721_hw4-data/ratings_test.csv",
                   header=None, names = column_names)


# In[3]:

print(data.shape)
print(data_test.shape)


# In[4]:

# initialize 
σ2 = 0.25
d = 10
λ = 1
#time = 10
iterate = 100

N1 = max(data['user_id'])  # 943
N2 = max(data['movie_id']) # 1682
zero_data1 = np.zeros(shape=(N1,N2))
zero_data2 = np.zeros(shape=(N1,N2))
M = pd.DataFrame(zero_data1)
index = pd.DataFrame(zero_data2)

# fill M with data in rating.csv 
for i in range(data.shape[0]):
    user = data['user_id'][i] -1
    movie = data['movie_id'][i] -1
    rating = data['rating'][i]
    M[movie][user] = rating 
    


# In[5]:

zero_data1


# In[6]:

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[j][i] != 0:
            index[j][i] = 1
    


# In[8]:

zero_data2


# In[13]:

# run 1
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[14]:

L1 = L
u1 = u
v1 = v

pred1 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred1.append(np.dot(u1[u_i-1,:],v1[:,v_j-1]))
print(time.clock()-start)
RMSE1 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[38]:

RMSE1 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[9]:

# run 2
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[10]:

L2 = L
u2 = u
v2 = v

pred2 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred2.append(np.dot(u2[u_i-1,:],v2[:,v_j-1]))
print(time.clock()-start)


# In[39]:

RMSE2 = ((pred2-data_test['rating'])**2).sum()/data_test.shape[0]


# In[15]:

# run 3
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[16]:

L3 = L
u3 = u
v3 = v

pred3 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred3.append(np.dot(u3[u_i-1,:],v3[:,v_j-1]))
print(time.clock()-start)


# In[40]:

RMSE3 = ((pred3-data_test['rating'])**2).sum()/data_test.shape[0]


# In[17]:

# run 4
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[18]:

L4 = L
u4 = u
v4 = v

pred4 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred4.append(np.dot(u4[u_i-1,:],v4[:,v_j-1]))
print(time.clock()-start)
RMSE4 = ((pred4-data_test['rating'])**2).sum()/data_test.shape[0]


# In[41]:

RMSE4 = ((pred4-data_test['rating'])**2).sum()/data_test.shape[0]


# In[19]:

# run 5
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[20]:

L5 = L
u5 = u
v5 = v

pred5 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred5.append(np.dot(u5[u_i-1,:],v5[:,v_j-1]))
print(time.clock()-start)
RMSE5 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[42]:

RMSE5 = ((pred5-data_test['rating'])**2).sum()/data_test.shape[0]


# In[21]:

# run 6
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[22]:

L6 = L
u6 = u
v6 = v

pred6 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred6.append(np.dot(u6[u_i-1,:],v6[:,v_j-1]))
print(time.clock()-start)
RMSE6 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[43]:

RMSE6 = ((pred6-data_test['rating'])**2).sum()/data_test.shape[0]


# In[23]:

# run 7
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[24]:

L7 = L
u7 = u
v7 = v

pred7 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred7.append(np.dot(u7[u_i-1,:],v7[:,v_j-1]))
print(time.clock()-start)
RMSE7 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[44]:

RMSE7 = ((pred7-data_test['rating'])**2).sum()/data_test.shape[0]


# In[25]:

# run 8
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[26]:

L8 = L
u8 = u
v8 = v

pred8 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred8.append(np.dot(u8[u_i-1,:],v8[:,v_j-1]))
print(time.clock()-start)
RMSE8 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[45]:

RMSE8 = ((pred8-data_test['rating'])**2).sum()/data_test.shape[0]


# In[27]:

# run 9
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[28]:

L9 = L
u9 = u
v9 = v

pred9 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred9.append(np.dot(u9[u_i-1,:],v9[:,v_j-1]))
print(time.clock()-start)
RMSE9 = ((pred1-data_test['rating'])**2).sum()/data_test.shape[0]


# In[46]:

RMSE9 = ((pred9-data_test['rating'])**2).sum()/data_test.shape[0]


# In[29]:

# run 10
u = np.random.normal(loc=0, scale=1, size=(N1,d))
v = np.random.normal(loc=0, scale=1, size=(d,N2))
import time
start = time.clock()
L = [0] * iterate
for itera in range(iterate):  #iterate

    # update u
    for i in range(N1):
        m_j = [i for i, e in enumerate(zero_data1[i,:]) if e != 0]
        vv = np.dot(v[:,m_j],v[:,m_j].T)
        Mv = (zero_data1[i,m_j]*v[:,m_j]).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+vv)
        u[i,:]= np.dot(p1,Mv.T)

    # update v
    for j in range(N2):
        # calculate Σuu
        u_i = [i for i, e in enumerate(zero_data1[:,j]) if e != 0]
        uu = np.dot(u[u_i,:].T,u[u_i,:])
        Mu = (zero_data1[u_i,j]*u[u_i,:].T).sum(axis=1)

        # create a diagonal matrix
        dia = np.zeros((d, d), float)
        np.fill_diagonal(dia, λ*σ2)
        p1 = inv(dia+uu)

        v[:,j]= np.dot(p1,Mu)

    p1 = ((zero_data1 - u.dot(v)*zero_data2)**2).sum()/(2*σ2)
    p2 = (u**2).sum()*λ/2
    p3 = (v**2).sum()*λ/2

    L[itera] = -p1-p2-p3
    
print(time.clock()-start)


# In[30]:

L10 = L
u10 = u
v10 = v

pred10 = []
start = time.clock()
for n in range(data_test.shape[0]):
    u_i = data_test['user_id'][n]
    v_j = data_test['movie_id'][n]
    pred10.append(np.dot(u10[u_i-1,:],v10[:,v_j-1]))
print(time.clock()-start)
RMSE10 = ((pred10-data_test['rating'])**2).sum()/data_test.shape[0]


# In[47]:

RMSE10 = ((pred10-data_test['rating'])**2).sum()/data_test.shape[0]


# In[48]:

import matplotlib.pyplot as plt
r1 = plt.scatter(x=np.arange(2,101,1), y=L1[1:100], s=20)
r2 = plt.scatter(x=np.arange(2,101,1), y=L2[1:100], s=20)
r3 = plt.scatter(x=np.arange(2,101,1), y=L3[1:100], s=20)
r4 = plt.scatter(x=np.arange(2,101,1), y=L4[1:100], s=20)
r5 = plt.scatter(x=np.arange(2,101,1), y=L5[1:100], s=20)
r6 = plt.scatter(x=np.arange(2,101,1), y=L6[1:100], s=20)
r7 = plt.scatter(x=np.arange(2,101,1), y=L7[1:100], s=20)
r8 = plt.scatter(x=np.arange(2,101,1), y=L8[1:100], s=20)
r9 = plt.scatter(x=np.arange(2,101,1), y=L9[1:100], s=20)
r10 = plt.scatter(x=np.arange(2,101,1), y=L10[1:100], s=20)
plt.legend((r1,r2,r3,r4,r5,r6,r7,r8,r9,r10),
           ('run1','run2','run3','run4','run5','run6','run7','run8','run9','run10'),
           scatterpoints=1,
           loc='lower right',
           ncol=2,
           fontsize=10)
plt.xlabel("Iterations",fontsize=12)
plt.ylabel("Log Joint Likelihood",fontsize=12)
plt.show()


# In[53]:

ljl = [L1[99], L2[99], L3[99], L4[99], L5[99], L6[99], L7[99], L8[99], L9[99], L10[99]]
RMSE = [RMSE1**.5, RMSE2**.5, RMSE3**.5, RMSE4**.5, RMSE5**.5, RMSE6**.5, RMSE7**.5, RMSE8**.5, RMSE9**.5, RMSE10**.5]
table = pd.DataFrame({'Log Joint Likelihood': ljl,
                      'RMSE': RMSE})
table


# In[54]:

table.sort(['Log Joint Likelihood'], ascending=False)


# In[62]:

sw = 50
fl = 485
gf = 182
names =  pd.read_table("/Users/sihui/Desktop/COMS4721ML/HW/HW4/COMS4721_hw4-data/movies.txt",
                   header=None)


# In[73]:

ed_sw = (((v7-v7[:,sw-1].reshape(10, 1))**2).sum(axis=0))**.5
sort_index = np.argsort(ed_sw)[1:11]
ed_sw[sort_index]
table1 = pd.DataFrame({'10 closest movies to Star Wars': names.iloc[sort_index].values.reshape(10,),
                      'Euclidean Distance': ed_sw[sort_index]})
table1


# In[84]:

ed_fl = (((v7-v7[:,fl-1].reshape(10, 1))**2).sum(axis=0))**.5
sort_index = np.argsort(ed_fl)[1:11]
ed_fl[sort_index]
table2 = pd.DataFrame({'10 closest movies to My Fair Lady': names.iloc[sort_index].values.reshape(10,),
                      'Euclidean Distance': ed_fl[sort_index]})
table2


# In[85]:

ed_gf = (((v7-v7[:,gf-1].reshape(10, 1))**2).sum(axis=0))**.5
sort_index = np.argsort(ed_gf)[1:11]
ed_gf[sort_index]
table3 = pd.DataFrame({'10 closest movies to Goodfellas': names.iloc[sort_index].values.reshape(10,),
                      'Euclidean Distance': ed_gf[sort_index]})
table3





