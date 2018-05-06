# python: 3.5.2
# encoding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from cvxopt import matrix, solvers
import time
import pandas as pd

def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    length=len(label)
    correctCount=0
    for i in range(length):
        if label[i]==pred[i]:
            correctCount+=1
    #用np.sum()好像会算两遍 所以换了一种方法算
    #return np.sum(label == pred) / len(pred)
    return correctCount/length

#计算支持向量和test数据kernel值
def calcu_kernel_value(X,Y,kTup):
    if kTup[0]=='linear':
        m,n=np.shape(X)
        kValue=np.zeros([m,1])
        d=kTup[1]
        for j in range(m):
            kValue[j]=np.dot(Xi,Xj)
    if kTup[0]=='poly':
        m,n=np.shape(X)
        kValue=np.zeros([m,1])
        d=kTup[1]
        for j in range(m):
            kValue[j]=pow(np.dot(X[j,:],Y)+1,d)
    if kTup[0]=='rbf':
        m,n=np.shape(X)
        kValue=np.zeros([m,1])
        sigma=kTup[1]
        for j in range(m):
            diff=X[j,:]-Y
            kValue[j]=np.exp(np.dot(diff,diff.T)/-2*sigma**2)
    return kValue

#计算kernel矩阵
def calcu_kernel_matrix(X,kTup):
    m,n=np.shape(X)
    kMatrix=np.zeros([m,m])
    for i in range(m):
        kValue=calcu_kernel_value(X,X[i,:],kTup)
        np.insert(kMatrix,i,values=kValue,axis=1)
    return kMatrix

#计算k(Xi,Xj)的值
def k(Xi,Xj,kTup): #2 columns
    if kTup[0]=='linear':
        value=np.dot(Xi,Xj)
    if kTup[0]=='poly':
        d=kTup[1]
        value=pow(np.dot(Xi,Xj)+1,d)
    if kTup[0]=='rbf':
        sigma=kTup[1]
        diff=Xi-Xj
        value=np.exp(np.dot(diff,diff.T)/-2*sigma**2)
    return value

#计算alpha
#使用cvxopt的solvers.qp计算二次规划problem
#C is the upper bound of alpha
def calcuAlpha(x_train,y_train,kTup):
    n=x_train.shape[0]
    #初始化
    #calculate coefficient Q, for the quadratic part
    #use k(Xi,Xj) to calculate the value of kernel 
    Q=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Q[i][j]=y_train[i]*y_train[j]*k(x_train[i,:],x_train[j,:],kTup)
    Q=matrix(Q,(n,n),'d')

    #calculate coefficient p for the linear part
    p=np.ones(n)
    p=-1*p
    p=matrix(p,(n,1),'d')
    
    #calculate the coefficient regarding to conditions: a1*t1+a2*t2+...+ai*ti+...=0
    A=matrix(y_train,(1,n),'d')
    b=matrix(0.0)

    #calculate the coefficient: -ai<=0
    G=-1*np.eye(n,n)
    G=matrix(G,(n,n),'d')
    h=np.zeros(n)
    h=matrix(h,(n,1),'d')

    sol=solvers.qp(Q, p, G, h, A, b)
    alpha=sol['x']

    #convert the min value to exact zero
    alpha=np.mat(alpha)
    for i in range(n):
            if alpha[i][0]<=1e-5:
                alpha[i][0]=0.0
    
    return alpha

def calcuB(x_train,y_train,alpha,kTup):
    X=np.mat(x_train)
    labels=np.mat(y_train).transpose()
    m,n=np.shape(X)
    w=np.zeros([n,1])
    b=0.0

    #get the support vector
    supportVectorsIndex=np.nonzero(alpha)[0]
    supportVectors=x_train[supportVectorsIndex]
    supportVectorLabels=y_train[supportVectorsIndex]
    supportAlpha=alpha[supportVectorsIndex]
    supportVectorLength=np.shape(supportAlpha)[0]

    #calculate b
    #pick one supportvector at r andom, here I always pickthe first supportvector
    for i in range(supportVectorLength):
        b+=supportAlpha[i]*supportVectorLabels[i]*k(supportVectors[i,:],supportVectors[0,:],kTup)
    b=supportVectorLabels[0]-b
    return b

def signX(x):
    if x<0:
        return -1
    elif x>0:
        return 1
    else:
        return 0

class SVM():
    """
    SVM模型。
    """
    def __init__(self, data_train,kTup):
        # 请补全此处代码
        self.x_train=data_train[:,:2]
        self.y_train=data_train[:,2]
        self.numSamples=self.x_train.shape[0] #初始行数
        self.b=0
        self.alpha=np.mat(np.zeros((self.numSamples,1)))
        self.kTup=kTup
        pass

    def train(self, data_train):
        """
        训练模型。
        """
        # 请补全此处代码
        self.alpha=calcuAlpha(self.x_train,self.y_train,self.kTup)
        self.b=calcuB(self.x_train,self.y_train,self.alpha,self.kTup)

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        x_test=x[:,:2]
        rowTest=x_test.shape[0]
        predict=np.zeros((rowTest,1))
        z=np.zeros((rowTest,1))
        alpha=self.alpha

        #support vector
        supportVectorsIndex=np.nonzero(alpha)[0]
        supportVectors=self.x_train[supportVectorsIndex]
        supportVectorLabels=self.y_train[supportVectorsIndex]
        supportVectorAlpha=alpha[supportVectorsIndex]
        supportVectorLength=np.shape(supportVectorAlpha)[0]  #the number of support vector
        print(supportVectorLength)
        tmp=np.mat(np.zeros((supportVectorLength,1)))

        for i in range(rowTest):
            kValue=calcu_kernel_value(supportVectors,x_test[i,:],self.kTup)
            for k in range(supportVectorLength):
                tmp[k]=supportVectorLabels[k]*supportVectorAlpha[k]
            z[i]=np.dot(kValue.T,tmp)+self.b
            predict[i]=signX(z[i])
        return z, predict

#画出SVM模型测试结果
def svmPlt(x_test,y_test, svm):
    X_set, y_set = x_test, t_test
    h=0.01
    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    
    plt.subplot(1,1,1)

    start=time.clock() # 开始计算训练模型耗费时间

    Z_set, Z_pred=svm.predict(np.c_[xx.ravel(),yy.ravel()])

    elapsed=(time.clock()-start) #结束计时
    print("Plting data time used:",elapsed)

    Z_set=Z_pred.reshape(xx.shape)
    plt.contourf(xx,yy,Z_set,cmap=ListedColormap(('red', 'green')),alpha=0.8)
    plt.scatter(X_set[:,0],X_set[:,1],c=y_set,cmap=ListedColormap(('orange', 'blue')))
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.xlim(xx.min(),xx.max())
    #plt.ylim(yy.min(),yy.max())
    plt.title('SVM Train')
    #plt.legend()
    plt.show()

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def gradAscent(X,Y,lamda=0,alpha=0.001,maxCycles=10000): #梯度下降法计算w
    xMatrix=np.mat(X)
    labelMatrix=np.mat(Y).transpose()
    m,n=np.shape(xMatrix)
    w=np.ones((n,1))
    for i in range(maxCycles):
        g=xMatrix*w
        h=sigmoid(g)
        error=h-labelMatrix
        G=(alpha/m)*(xMatrix.transpose()*error)+lamda/m*w
        w=w-G
    print(w)
    return w

def classify(x):
    if x>=0.5:
        return 1.0
    else:
        return -1.0

def logistic(x_train,y_train,lamda=0,alpha=0.001,maxCycles=1000000):
    w=gradAscent(x_train,y_train)
    numTest=np.shape(y_train)[0]

    def f(x): # calculate the predicting results
        xMatrix=np.mat(x)
        #print('x shape:',np.shape(x))
        #print('w shape:',np.shape(w))
        #print('x*w shape:',np.shape(x*w))
        g=xMatrix*w
        h=sigmoid(g)
        print(h.T)
        for i in range(numTest):
            h[i]=classify(h[i])
        return h
        pass

    return f

def linearPlot():
    return 0


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    #train_file='data/train_kernel.txt'
    #test_file='data/test_kernel.txt'

    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'

    #data_train_kernel=load_data(train_file_kernel)
    #data_test_kernel=load_data(test_file_kernel)

    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    
    # 使用训练集训练SVM模型
    """
    svm = SVM(data_train, ('rbf',0.5))  # 初始化模型
    #svm.train_kernel(data_train_kernel)
    
    start=time.clock() # 开始计算训练模型耗费时间

    svm.train(data_train)  # 训练模型

    elapsed=(time.clock()-start) #结束计时
    print("Model training time used:",elapsed)

    # 使用SVM模型预测标签
    start=time.clock() # 开始计算训练模型耗费时间
    
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    z_train, t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    z_test, t_test_pred = svm.predict(x_test)

    elapsed=(time.clock()-start) #结束计时
    print("Data predicting time used:",elapsed)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("SVM train accuracy: {:.1f}%".format(acc_train * 100))
    print("SVM test accuracy: {:.1f}%".format(acc_test * 100))

    svmPlt(x_test,y_test,svm)
    """

    #使用训练集训练logistic regression模型
    x_train = data_train[:, :2]  # feature [x1, x2]
    y_train = data_train[:, 2]  # 真实标签

    x_test = data_test[:, :2]
    y_test = data_test[:, 2]

    f=logistic(x_train,y_train)

    y_linear_train_pred = f(x_train)
    print(y_train.T)
    print(y_linear_train_pred.T)
    y_linear_test_pred = f(x_test)

    # 评估结果，计算准确率
    acc_linear_train = eval_acc(y_train, y_linear_train_pred)
    acc_linear_test = eval_acc(y_test, y_linear_test_pred)
    print("linear train accuracy: {:.1f}%".format(acc_linear_train * 100))
    print("linear test accuracy: {:.1f}%".format(acc_linear_test * 100))
    


    
