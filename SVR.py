import numpy as np
import cvxopt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def linear_kernel(x,y,c=0):
    return np.dot(x,y.T)

def poly_kernel(x,y,d):
    return (1+(np.dot(x,y.T))**d)
    
def gaussian_kernel(x,y,gamma):
    return np.exp(-(np.linalg.norm((x.T-y.T))**2)*gamma)

def rsquared_score(Y_test,Y_pred):
    data_var = np.sum((Y_test-np.mean(Y_test,axis=0))**2)
    model_var = np.sum((Y_pred-Y_test)**2)
    expl_var = data_var-model_var
    return float(expl_var/data_var)

def MSE(pred, Y_test):
    return np.sum(np.square(pred-Y_test))/Y_test.shape[0]

def MinMaxScaler(x_train,x_test):
    minimum = np.min(x_train, axis=0)
    maximum = np.max(x_train,axis= 0)
    x_train = (x_train-minimum)/(maximum-minimum)
    x_test = (x_test-minimum)/(maximum-minimum)
    return x_train,x_test

#THIS IS THE CODE FOR EPSILON SVR
def epsilon_svr(X_train,Y_train,X_test,c,kernel, kernel_param,epsilon):
    m, n = X_train.shape
    #Finding the kernels i.e. k(x,x')
    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernel(X_train[i,:], X_train[j,:], kernel_param)
    #Making matrices P,q,A,b
    element1 = k
    P= np.concatenate((element1,-1*element1),axis=1)
    P= np.concatenate((P,-1*P),axis=0)
    q= epsilon*np.ones((2*m,1))
    qadd=np.concatenate((-1*Y_train,Y_train),axis=0)
    q=q+qadd
    A=np.concatenate((np.ones((1,m)),-1*(np.ones((1,m)))),axis=1)
    #define matrices for optimization problem       
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.zeros((1,1)))
    #Making matrices G,h
    c= float(c)
    tem3=np.concatenate((np.eye(2*m),-1*np.eye(2*m)),axis=0)
    G=cvxopt.matrix(tem3)
    tem4=np.concatenate((c*np.ones((2*m,1)),np.zeros((2*m,1))),axis=0)
    h = cvxopt.matrix(tem4)
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    #getting the lagrange multipliers
    l = np.ravel(sol['x'])
    #parting to get the 2 sets of Lagrange multipliers
    u=l[0:m]
    v=l[m:]
    #Getting the  support vectors
    u1=u > 1e-5
    v1=v > 1e-5
    SV=np.logical_or(u1, v1)
    SVindices = np.arange(len(l)/2)[SV] 
    u1=u[SVindices.astype(int)]
    v1=v[SVindices.astype(int)]
    support_vectors_x = X_train[SV]
    support_vectors_y = Y_train[SV]
    #calculate intercept
    bias= sol['y']
    #find weight vector and predict y
    Y_pred = np.zeros((len(X_test),1))
    for i in range(len(X_test)):
        val=0
        for u_,v_,z in zip(u1,v1,support_vectors_x):
            val=val+(u_ - v_)*kernel(X_test[i],z,kernel_param)
        Y_pred[i,0]= val
    Y_pred = Y_pred+bias[0,0]
    return Y_pred

#THIS IS THE CODE FOR RHSVR
def rh_svr(X_train,Y_train,X_test,c,kernel, kernel_param,epsilon):
    #Scaling Y since  RHSVR essentitally solves a classfication problem with y+-epsilon as one of the features
    Y_train_max=np.max(Y_train,axis=0)
    Y_train_min=np.min(Y_train,axis=0)
    Y_train=(Y_train-Y_train_min)/(Y_train_max-Y_train_min)
    #Finding number of features and data points
    m, n = X_train.shape
    #Finding the kernels i.e. k(x,x')
    k = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            k[i][j] = kernel(X_train[i,:], X_train[j,:], kernel_param)
    #Making the matrices P,q,A,b
    element1 = k+np.dot(Y_train,Y_train.T)
    Prow= np.concatenate((element1,-1*element1),axis=1)
    P= np.concatenate((Prow,-1*Prow),axis=0)
    q= 2*epsilon*np.concatenate((Y_train,-1*Y_train),axis=0)
    tem1=np.concatenate((np.ones((1,m)),np.zeros((1,m))),axis=1)
    tem2=np.concatenate((np.zeros((1,m)),np.ones((1,m))),axis=1)
    A=np.concatenate((tem1,tem2),axis=0)
    #define matrices for optimization problem    
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.ones((2,1)))
    #Making matrices G,h
    c= float(c)
    tem3=np.concatenate((np.eye(2*m),-1*np.eye(2*m)),axis=0)
    G=cvxopt.matrix(tem3)
    tem4=np.concatenate((c*np.ones((2*m,1)),np.zeros((2*m,1))),axis=0)
    h = cvxopt.matrix(tem4)
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    #getting the lagrange multipliers
    l = np.ravel(sol['x'])
    #parting to get the 2 sets of Lagrange multipliers
    u_cap=l[0:m]
    v_cap=l[m:]
    #getting the scaled lagrange multifliers
    delta=np.dot((u_cap-v_cap).T,Y_train)+2*epsilon
    u_bar=u_cap/delta
    v_bar=v_cap/delta
    #getting the support vectors
    u1=u_bar > 1e-5
    v1=v_bar > 1e-5
    SV=np.logical_or(u1, v1)
    SVindices = np.arange(len(l)/2)[SV]
    u=u_bar[SVindices.astype(int)]
    v=v_bar[SVindices.astype(int)]
    support_vectors_x = X_train[SV]
    support_vectors_y = Y_train[SV]
    #calculating the intercept
    bias= np.dot((u_cap-v_cap).T,k)
    bias= np.dot(bias,(u_cap+v_cap))
    bias = bias/(2*delta)+np.dot((u_cap+v_cap).T,Y_train)/2
    #calculating the hyperplane parameters w & b & d : w_cap.T*x+delta*y+b_cap=0
    w_cap=np.dot(X_train.T,u_cap-v_cap)
    delta_cap=delta
    b_cap= -1*(np.dot(w_cap.T,np.dot(X_train.T,(u_cap+v_cap)))/2+delta_cap*np.dot(Y_train.T,(u_cap+v_cap))/2)
    #predicting Y
    Y_pred = np.zeros((len(X_test),1))
    for i in range(len(X_test)):
        val=0
        for u_,v_,z in zip(u,v,support_vectors_x):
            val+=(v_ -u_)*kernel(X_test[i],z,kernel_param)
        Y_pred[i,0]= val
    Y_pred = Y_pred+bias
    return Y_pred*(Y_train_max-Y_train_min)+Y_train_min #scaling back predictions

#THIS IS THE CODE FOR SKLEARN BASED SVR
def sklearn_svr(X_train, Y_train,X_test, kernel_type, eps,reg_param, kernel_param):  
    if(kernel_type ==1 ):
        regressor = SVR(kernel='linear', epsilon=eps, C=reg_param)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    elif(kernel_type == 2):
        regressor = SVR(kernel='poly', epsilon=eps,degree=kernel_param, C=reg_param,gamma=1.0)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    elif(kernel_type == 3) :
        regressor= SVR(kernel = 'rbf', epsilon = eps, C= reg_param, gamma=kernel_param)
        regressor.fit(X_train,Y_train)
        y_pred=regressor.predict(X_test)
    return y_pred

#THIS IS THE CODE FOR COMPARING TWO MODELS
def compare_model(X,Y,model,kernel,ker_index,epsilon,C,D,kernel_param): #ker_index is 1 for linear, 2 for poly, 3 for rbf
    kf = KFold(n_splits = 5,random_state=None, shuffle = False)
    scores_1 = []
    mse_scores1=[]
    scores_2  = []
    mse_scores2=[]
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index].reshape((len(X_test),1))
        #scale the data by min-max scaler
        X_train,X_test = MinMaxScaler(X_train,X_test)
        #get predictions from cvxopt model
        pred=model(X_train,Y_train,X_test,C,kernel,kernel_param,epsilon)#REPLACE D BY C FOR EPSILON SVR
        scores_1.append(rsquared_score(Y_test,pred))
        mse_scores1.append(MSE(Y_test,pred))
        #get predictions from sklearn model
        y_pred = sklearn_svr(X_train,Y_train,X_test, ker_index, epsilon, C, kernel_param)
        scores_2.append(r2_score(Y_test,y_pred))
        mse_scores2.append(mean_squared_error(Y_test,y_pred))
    print("MSE")
    print("RHcvxopt===>"+str(np.around((mse_scores1),decimals=5)))
    print("sklearn===>" +str(np.around((mse_scores2), decimals=5)))
    print("R2")
    print("RHcvxopt===>"+str(np.around((scores_1),decimals=5)))
    print("sklearn===>" +str(np.around((scores_2), decimals=5)))
    return float(np.around(np.mean(scores_1),decimals=5)),float(np.around(np.mean(scores_2),decimals=5)),float(np.around(np.mean(mse_scores1),decimals=5)),float(np.around(np.mean(mse_scores2),decimals=5))

#IMPORTING DATA
data=pd.read_csv('housing.csv', header=None,delimiter=r"\s+")
numpydata=data.values
X= numpydata[:,:13]
Y= numpydata[:,13:14]

#PLOTTING GRAPHS: vals gives the range of the variabel you wish to loop over for getting a graph
vals =np.arange(1,3,0.2)
r1=[]
r2=[]
a1=[]
a2=[]
for epsilon in vals:  #put the variable here which needs to be varied for the graph
    t1,t2,m1,m2=compare_model(X,Y,epsilon_svr,linear_kernel,1 ,epsilon, 1,1, 2) #(X,Y,MODEL,KERNEL,NUMBER,EPSILON,C,D,KERNEL_PARAM) C=D FOR EPSILON VS SKLEARN
    r1.append(t1)
    r2.append(t2)
    a1.append(m1)
    a2.append(m2)
print(r1)
print(a1)    
plt.figure()
plt.plot(vals,r1,label="RH SVR GAUSSIAN", marker='o')
plt.plot(vals,r2,label="SKLearn LINEAR", marker='x')
plt.xlabel("D")
plt.ylabel("R-squared coeff")
plt.title('gamma=0.3 & epsilon=0.0175')
plt.legend()
plt.savefig('d1')
plt.show()        
        
plt.figure()
plt.plot(vals,a1,label="RH SVR GAUSSIAN", marker='o')
plt.plot(vals,a2,label="SKLearn LINEAR", marker='x')
plt.xlabel("D")
plt.ylabel("MSE")
plt.title('gamma=0.3 & epsilon=0.0175')
plt.legend()
plt.savefig('d2')
plt.show()

