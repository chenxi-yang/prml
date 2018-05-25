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
import bigfloat

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

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def gradAscent(X,Y,lamda,alpha,maxCycles,modelType): #梯度下降法计算w
    #xMatrix=np.mat(np.c_[np.ones((np.shape(X)[0],1)),X])
    labelMatrix=Y
    m,n=np.shape(X)
    print(n)
    w=np.array(np.zeros(n))
    error=np.zeros((m,1))
    for i in range(maxCycles):
        #print(w.reshape(-1,1))
        g=X.dot(w.reshape(-1,1))
        if modelType=='linear': #对损失函数求导后 得2(yt-t)x
        	h=g 
        	for k in range(m):
        		if h[k]>0:
        			h[k]=1
        		elif h[k]<0:
        			h[k]=-1
        	error=h-labelMatrix
        if modelType=='logistic':
        	h=sigmoid(g)
        	error=h-(labelMatrix==1)#由于logistic函数算出来的是[0,1], 而分类为1，-1，因此把-1转换为0
        if modelType=='hinge':
        	h=np.multiply(g,labelMatrix) #t*y
        	for k in range(m):
        		if h[k]<1:
        			error[k]=-labelMatrix[k]
        		#elif h[k]<1:
        		#	error[k]=-labelMatrix[k]*(1-h[k])
        		else:
        			error[k]=0
        E=(1.0/m)*((error.T).dot(X))+lamda/m*(np.r_[[0],w[1:]])
        w=w-alpha*E.flatten()
    return w

def sign2(x,modelType):
	if modelType=='logistic':
		condition=0.5
	else:
		condition=0.0
	if x>=condition:
		return 1
	else:
		return -1

def classify(x_train,y_train,lamda,alpha,maxCycles,modelType):
    x=np.c_[np.ones((np.shape(x_train)[0],1)),x_train]
    #print(np.shape(x))
    y=np.c_[y_train]
    w=gradAscent(x,y,lamda,alpha,maxCycles,modelType)
    numTest=np.shape(y)[0]
    #print(w)

    def f(x): # calculate the predicting results
        #print(np.shape(x))
        xMatrix=np.mat(np.c_[np.ones((np.shape(x)[0],1)),x])
        #print(np.shape(xMatrix))
        g=xMatrix.dot(w.reshape(-1,1))
        if modelType=='logistic':
        	h=sigmoid(g)
        else:
        	h=g
        for i in range(numTest):
            h[i]=sign2(h[i],modelType)
        return h.astype('int')
        pass

    return w,f,modelType

#画出线性二分类结果的图
def linearPlotData(x,axes=None):
    neg_data=(x[:,2]==-1)
    pos_data=(x[:,2]==1)
    if axes==None:
        axes=plt.gca()
    axes.scatter(x[neg_data][:,0],x[neg_data][:,1],s=40,c='red',marker='o',label=-1)
    axes.scatter(x[pos_data][:,0],x[pos_data][:,1],s=40,c='green',marker='x',label=1)
    axes.set_xlabel('X0')
    axes.set_ylabel('X1')
    axes.legend(frameon=True,fancybox=True)
    
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
    """
    #使用训练集训练logistic regression模型
    
    x_train = data_train[:, :2]  # feature [x1, x2]
    y_train = data_train[:, 2]  # 真实标签

    x_test = data_test[:, :2]
    y_test = data_test[:, 2]

    start=time.clock() # 开始计算训练模型耗费时间
    w,f,modelType=classify(x_train,y_train,lamda=10,alpha=0.002,maxCycles=1000,modelType='linear')
    elapsed=(time.clock()-start) #结束计时
    print("Classification time:",elapsed)

    y_linear_train_pred = f(x_train)
    y_linear_test_pred = f(x_test)

    # 评估结果，计算准确率
    acc_linear_train = eval_acc(y_train, y_linear_train_pred)
    acc_linear_test = eval_acc(y_test, y_linear_test_pred)
    print("linear train accuracy: {:.1f}%".format(acc_linear_train * 100))
    print("linear test accuracy: {:.1f}%".format(acc_linear_test * 100))
    """    
    linearPlotData(data_train)
    """
    h=0.1
    x_min,x_max=x_test[:,0].min(),x_test[:,0].max()
    y_min,y_max=x_test[:,1].min(),x_test[:,1].max()
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    if modelType=='logistic':
    	h=sigmoid(np.c_[np.ones((xx.ravel().shape[0],1)),xx.ravel(),yy.ravel()].dot(w.reshape(-1,1)))
    else:
    	h=np.c_[np.ones((xx.ravel().shape[0],1)),xx.ravel(),yy.ravel()].dot(w.reshape(-1,1))
    h=h.reshape(xx.shape)

    plt.contour(xx,yy,h,[0.5],linewidths=1,colors='r')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    """
    plt.title('Linear Train Data')
    plt.show()
    
    #使用训练集训练logistic regression模型



    
