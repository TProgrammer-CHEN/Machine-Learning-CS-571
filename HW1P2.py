# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:10:18 2018

@author: Iven
"""
from numpy import linalg as LA;import numpy as np;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd;
from mnist import MNIST;
from sklearn.metrics import confusion_matrix

#1(a) of part 1
#y=sign(x1-x2+0.5)
plt.scatter([0,0,1,1],[0,1,0,1],c=(0,1,0,0));plt.plot([0,1],[0.5,1.5])

#1(b) of part 1
plt.scatter([0,0,1,1],[0,1,0,1],c=(0,1,1,0))

#1(c) of part 1
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 1, 0.05)
Y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(X, Y)
R = np.sqrt((X + 2*Y)/3+0.1)
Z = np.sin(R)

# Plot the surface.
ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False)
# Customize the z axis.
ax.scatter(xs=[0,0,0,0,1,1,1,1],ys=[0,0,1,1,0,0,1,1],zs=[0,1,0,1,0,1,0,1])
plt.show()


mndata = MNIST('samples',gz=True)
trainx, trainy = mndata.load_training()
testx, testy = mndata.load_testing()

#transform data into (-1,1) labels
trdata=np.array([np.array(trainx[i]) for i in range(len(trainy)) if trainy[i] in [4,9]])
trlabel=np.array([2*(trainy[i]==4)-1 for i in range(len(trainy)) if trainy[i] in [4,9]])
tstdata=np.array([np.array(testx[i]) for i in range(len(testy)) if testy[i] in [4,9]])
tstlabel=np.array([2*(testy[i]==4)-1 for i in range(len(testy)) if testy[i] in [4,9]])

#normalize train data by the maxium 2 norm of samples
def normalize(x):
    normalizer=np.amax([LA.norm(samp) for samp in x])
    xnew= x/normalizer
    return xnew
trdata=normalize(trdata)
tstdata=normalize(tstdata)

    

#function of Perceptron

class perceptron():
    def __init__(self,x,y):
        self.x=x;self.y=y
        self.size=len(x)
        self.dim=len(x[0])
        self.w=np.zeros(self.dim)
        self.wlist=[self.w]
# fit the model on traindata   
    def fit(self,epo):
        w=np.zeros(self.dim);wlist=[w];
        corrc=[0]
        for e in range(epo):
            for i in range(self.size):
                samp=self.x[i]
                val=self.y[i]*(sum(np.multiply(samp,w)))
                if (val<=0):
                    tmp=[self.y[i]*samp[j] for j in range(self.dim)]
                    w=np.add(w,tmp)
            wlist=np.append(wlist,[w],axis=0)
            pred=[np.sign(sum(np.multiply(w,self.x[i]))) for i in range(self.size)]
            corrc=np.append(corrc,sum(np.equal(self.y,pred))/self.size)
        self.w=w
        self.wlist=wlist
        self.corrc=corrc
# test the perceptron on testdata   
    def test(self,testx,testy):
        itr=len(self.wlist)
        tsize=len(testy)
        corrc=[]
        for e in range(itr):
            pred=[np.sign(sum(np.multiply(self.wlist[e],testx[i]))) for i in range(tsize)]
            corrc=np.append(corrc,sum(np.equal(testy,pred))/tsize)
        return corrc
# output the confusion matix    
    def confusion(self,testx,testy):
        tsize=len(testy)
        w=self.w
        pred=[np.sign(sum(np.multiply(w,testx[i]))) for i in range(tsize)]
        corrc=sum(np.equal(testy,pred))/tsize
        tn, fp, fn, tp=confusion_matrix(testy, pred).ravel()
        return corrc,tn,fp,fn,tp
# output the w' and compute the sequence of TPR and FPR       
    def dfit(self):
        onethird=round(self.size/3)
        w=np.zeros(self.dim)
        for i in np.arange(onethird):
            samp=self.x[i]
            val=self.y[i]*(sum(np.multiply(samp,w)))
            if (val<=0):
                    tmp=[self.y[i]*samp[j] for j in range(self.dim)]
            w=np.add(w,tmp)
        wt=w
        wstar=self.w
        #ROC curve
        pt=[];nt=[];pstar=[];nstar=[]
        for e in np.arange(-30,30,0.1):
            tn,fp,fn,tp=confusion_matrix(self.y,[np.sign(sum(np.multiply(wstar,self.x[i]))+e) for i in range(len(self.y))]).ravel()
            pstar=np.append(pstar,tp/(tp+fn))
            nstar=np.append(nstar,fp/(fp+tn))
            
            tn,fp,fn,tp=confusion_matrix(self.y,[np.sign(sum(np.multiply(wt,self.x[i]))+e) for i in range(len(self.y))]).ravel()
            pt=np.append(pt,tp/(tp+fn))
            nt=np.append(nt,fp/(fp+tn))
        
        return pstar,nstar,pt,nt

    def auc(self,tpr,fpr):
        res=np.trapz(tpr,x=fpr)
        return res

pr=perceptron(trdata,trlabel)
pr.fit(100)
plt.plot(np.arange(0,101),pr.corrc)

c=pr.test(tstdata,tstlabel)
plt.plot(np.arange(0,101),c)

c,tn,fp,fn,tp=pr.confusion(tstdata,tstlabel)

tpr1,fpr1,tpr2,fpr2=pr.dfit()
plt.plot(fpr1,tpr1);plt.plot(fpr2,tpr2)

np.trapz(tpr1,x=fpr1)
np.trapz(tpr2,x=fpr2)
  
class winnow:
    def __init__(self,x,y):
        self.x=x;self.y=y
        self.size=len(x)
        self.dim=len(x[0])
        self.wp=np.full(self.dim,1/self.dim)
        self.wn=np.full(self.dim,1/self.dim)
   # fit the model on train data and output weights
    def fit(self,epo,eta):
        wp=np.full(self.dim,1/self.dim)
        wn=np.full(self.dim,1/self.dim)
        corrc=[0]
        for e in range(epo):
            for i in range(self.size):
                samp=self.x[i]
                val=self.y[i]*(sum(np.multiply(samp,wp))-sum(np.multiply(samp,wn)))
                if (val<=0):
                    wp=np.multiply(wp,np.exp(eta*self.y[i]*self.x[i]))
                    wn=np.multiply(wn,np.exp(-eta*self.y[i]*self.x[i]))
                    s=sum(wp)+sum(wn)
                    wp=wp/s
                    wn=wn/s
            pred=[np.sign(sum(np.multiply(wp,self.x[i]))-sum(np.multiply(wn,self.x[i]))) for i in range(self.size)]
            corrc=np.append(corrc,sum(np.equal(self.y,pred))/self.size)
        self.wp=wp
        self.wn=wn
        self.corrc=corrc
        return corrc
    # test the performance of model on test data
    def test(self,testx,testy):
        tsize=len(testy)

        pred=[np.sign(sum(np.multiply(testx[i],self.wp))-sum(np.multiply(testx[i],self.wn))) for i in range(tsize)]
        corrc=sum(np.equal(testy,pred))/tsize
        tn, fp, fn, tp=confusion_matrix(testy, pred).ravel()
        return corrc,tn,fp,fn,tp
    
wiw=winnow(trdata,trlabel)
wiw.fit(50,1)
#plot evolution of accuracies
plt.plot(np.arange(0,31),wiw.corrc)
#output the corrections and the confusion matrix data
wiw.test(tstdata,tstlabel)
