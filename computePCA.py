#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:14:38 2021

@author: marian & elisa
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

def generateFile(label,Y,dataFile):
    att=['Y1','Y2']+label
    f=open('csv/pca.csv','w')
    fin=open(dataFile)
    for i in range(len(att)-1):
        print(att[i],',',end='',file=f)
    print(att[-1],file=f)
    for i in range(len(Y)-1):
        s=fin.readline().strip()
        print(Y[i][0],',',Y[i][1],',',s,file=f)   
    f.close()
    

###read data from a CSV file, you can choose different delimiters
att=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
     'RestingECG','MaxHR	','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']

d=np.genfromtxt('csv/heart_modified_nonames.csv',skip_header=0,usecols=[i for i in range(0,12)],delimiter=',') 
#normalize the data with StandardScaler
d_std = preprocessing.StandardScaler().fit_transform(d)
#compute PCA
pca=PCA(n_components=11)
d_pca=pca.fit_transform(d_std)
#d_pca is a numpy array with transformed data and pca is a
# PCA variable  with useful attributes (e.g., explained_variance_)

generateFile(att,d_pca,'csv/heart_modified_nonames.csv')



plt.plot(d_pca[:,0],d_pca[:,1], 'o', markersize=3, color='blue', alpha=0.5, 
         label='PCA transformed data in the new 2D space')    
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed data from sklearn.decomposition import PCA')

plt.show()

s = 30
plt.scatter(d_pca[0:410, 0], d_pca[0:410, 1],
            color='#67a9cf',s=s, lw=0, label='Normal')
plt.scatter(d_pca[410:918, 0], d_pca[410:918, 1],
            color='#ef8a62',s=s, lw=0, label='Heart Disease')

plt.xlabel('Y1')
plt.ylabel('Y2')
plt.legend()
plt.title('Transformed data from sklearn.decomposition import PCA')

plt.show()

#-------------------------------------------------------------

d_cov=np.cov(d.T)
for i in range(len(d_cov)):
    print('Variance original data axis X'+str(i+1),d_cov[i][i])
print('Covariance matrix')    

for i in range (len(d_cov)):
    for j in range(len(d_cov[0])):
        print('%.2f ' % (d_cov[i][j]), end='\t')
        #print(str(d_pca[i][j])[:6]+' ', end='')
    print()
print('-------------------------------------')

#--------------------------------------------------------------

d_cov=np.cov(d_std.T)
for i in range(len(d_cov)):
    print('Variance original normalized data axis X'+str(i+1),d_cov[i][i])

print('Covariance matrix')  
for i in range (len(d_cov)):
    for j in range(len(d_cov[0])):
        print('%.2f ' % (d_cov[i][j]), end='\t')
        #print(str(d_pca[i][j])[:6]+' ', end='')
    print()    
print('-------------------------------------')

#--------------------------------------------------------------

d_cov=np.cov(d_pca.T)
for i in range(len(d_cov)):
    print('Variance transformed data axis Y'+str(i+1),d_cov[i][i])

print('Covariance matrix')
for i in range (len(d_cov)):
    for j in range(len(d_cov[0])):
        print('%.2f ' % (d_cov[i][j]), end='\t')
        #print(str(d_pca[i][j])[:6]+' ', end='')
    print()
print('-------------------------------------')

#--------------------------------------------------------------

#compute and sort eigenvalues
v=pca.explained_variance_ratio_
print('Cumulated variance of the first two PCA components:',
      (v[0]+v[1]))
