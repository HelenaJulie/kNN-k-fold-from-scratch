# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:00:47 2018

@author: Helena J Arpudaraj
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 10:37:08 2018

@author: Helena J Arpudaraj
"""
import struct
import numpy as np
import math
from statistics import mode
    
def read_idx(filename): 
    with open(filename, 'rb') as f: 
        zero, data_type, dims = struct.unpack('>HBB', f.read(4)) 
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims)) 
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape) 

    
raw_train=read_idx("train-images.idx3-ubyte")
train_data=np.reshape(raw_train,(60000,28*28))
train_label=read_idx("train-labels.idx1-ubyte")
#X = train_data(raw_trainlabel == 8)
#Y = manifold.Isomap(n_neighbors=5,n_components=2).fit_transform.X
raw_test=read_idx("t10k-images.idx3-ubyte")
test_data=np.reshape(raw_test,(10000,28*28))
test_label=read_idx("t10k-labels.idx1-ubyte")

#confusion matrix 
def confusion_matrix(predicted_array,actual_array):
    confusion_array=np.zeros((10,10))
    for i,j in zip(predicted_array,actual_array):
        confusion_array[i][j]+=1
    return confusion_array                
            
        
#knn for k=1
#finding euclidean distance
counter=0
EuclideanDist = np.zeros((10000,60000))
for i in range (10000):
    for j in range (60000):
        d=0.0              
        for k in range (784):
            d=d+float(pow((float(train_data[j,k])-float(test_data[i,k])),2))
        d=float(math.sqrt(d))
        EuclideanDist[i,j]=d                 
         
True_Table = np.zeros((10000,2))

#finding 1 nearest neighbor
for i in range (10000):
    A=np.array(EuclideanDist[i])
    x=int(A.argmin())
    True_Table[i,0]=test_label[i]
    True_Table[i,1]=train_label[x]

#finding accuracy for knn where k=1    
test=[]
predict=[]
for i in range (10000):
    test.append(int(True_Table[i,0]))
    predict.append(int(True_Table[i,1]))
    
x=0
for i in range (10000):
    if(True_Table[i,0]==True_Table[i,1]):
        x=x+1
accuracy=(x/10000)*100

print('knn accuracy for MNIST data set when k=1: ',accuracy)

print('')
#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict,test)
print(cm)
print('')

#10-fold validation
#setting the test and train data
#train_data_kfold=np.reshape(raw_train,(60000,28*28))
#train_label_kfold=read_idx("train-labels.idx1-ubyte")
#z = list(zip(train_data_kfold, train_label_kfold))
#random.shuffle(z)
#train_data_kfold, train_label_kfold=zip(*z)
NumofTrainData=60000
folds=int(NumofTrainData/10)
for i in range(10):
    kfoldTest_data=train_data[(folds*i):(folds*(i+1))]
    kfoldTest_label=train_label[(folds*i):(folds*(i+1))]
    if(i>0):
        kfoldTrain_Data=train_data[0:(folds*i)]
        kfoldTrain_label=train_label[0:(folds*i)]
        if (i<9):
            kfoldTrain_Data=np.append(kfoldTrain_Data,train_data[folds*(i+1):60000],axis=0)    
            kfoldTrain_label=np.append(kfoldTrain_label,train_label[folds*(i+1):60000],axis=0)    
    else:
        kfoldTrain_Data=train_data[folds*(i+1):60000] 
        kfoldTrain_label=train_label[folds*(i+1):60000] 
    
    counter=0
    EuclideanDist1 = np.zeros(((folds),(folds*9)))
    for a in range (folds):
        for b in range (folds*9):
            d=0.0  
            for c in range (784):
                d1=float(kfoldTrain_Data[b,c])
                d2=float(kfoldTest_data[a,c])
                d=d+((d1-d2)**2)
                if (d<0):
                    d=d*(-1)
                d=float(math.sqrt(d))
                EuclideanDist1[a,b]=d                 
         
    k1True_Table = np.zeros(((folds*10),2))
    k2True_Table = np.zeros(((folds*10),2))
    k3True_Table = np.zeros(((folds*10),2))
    k4True_Table = np.zeros(((folds*10),2))
    k5True_Table = np.zeros(((folds*10),2))
    k6True_Table = np.zeros(((folds*10),2))
    k7True_Table = np.zeros(((folds*10),2))
    k8True_Table = np.zeros(((folds*10),2))
    k9True_Table = np.zeros(((folds*10),2))
    k10True_Table = np.zeros(((folds*10),2))

    #finding k=1,2,...10 nearest neighbor
    for a in range (folds):
        B=np.array(EuclideanDist1[a])
        x=int(B.argmin())
        for j in range (1,11):
            if (j==1):
                k1True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                k1True_Table[((folds*i)+a),1]=kfoldTrain_label[x]
            else:
                kneigbors=[]
                kneigbors=B.argsort()[:j]
                kneighbor_label=[]
                for k in range(j):
                    kneighbor_label.append(kfoldTrain_label[kneigbors[k]])
                try:
                    label=mode(kneighbor_label)
                except:
                    label1=int(B.argmin())
                    label=train_label[label1]                       
                if (j==2):
                    k2True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k2True_Table[((folds*i)+a),1]=label
                elif (j==3):
                    k3True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k3True_Table[((folds*i)+a),1]=label 
                elif (j==4):
                    k4True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k4True_Table[((folds*i)+a),1]=label 
                elif (j==5):
                    k5True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k5True_Table[((folds*i)+a),1]=label 
                elif (j==6):
                    k6True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k6True_Table[((folds*i)+a),1]=label 
                elif (j==7):
                    k7True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k7True_Table[((folds*i)+a),1]=label 
                elif (j==8):
                    k8True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k8True_Table[((folds*i)+a),1]=label 
                elif (j==9):
                    k9True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k9True_Table[((folds*i)+a),1]=label 
                else:
                    k10True_Table[((folds*i)+a),0]=kfoldTest_label[a]
                    k10True_Table[((folds*i)+a),1]=label 


accuracy_list=[]
#finding accuracy for knn where k=1    
test1=[]
predict1=[]
x=0
for i in range (folds*10):
    if(k1True_Table[i,0]==k1True_Table[i,1]):
        x=x+1
    test1.append(int(k1True_Table[i,0]))
    predict1.append(int(k1True_Table[i,1]))
accuracy_list=(x/(folds*10))*100
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict1,test1)
print(cm)
print('')    

    
#finding accuracy for knn where k=2
test2=[]
predict2=[]

x=0
for i in range (folds*10):
    if(k2True_Table[i,0]==k2True_Table[i,1]):
        x=x+1
    test2.append(int(k2True_Table[i,0]))
    predict2.append(int(k2True_Table[i,1]))

accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')
#rows are predicted and columns are actual

print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict2,test2)
print(cm)
print('')    

#finding accuracy for knn where k=3
test3=[]
predict3=[]

x=0
for i in range (folds*10):
    if(k3True_Table[i,0]==k3True_Table[i,1]):
        x=x+1
    test3.append(int(k3True_Table[i,0]))
    predict3.append(int(k3True_Table[i,1]))

accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict3,test3)
print(cm)
print('')    

#finding accuracy for knn where k=4
test4=[]
predict4=[]

x=0
for i in range (folds*10):
    if(k4True_Table[i,0]==k4True_Table[i,1]):
        x=x+1
    test4.append(int(k4True_Table[i,0]))
    predict4.append(int(k4True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict4,test4)
print(cm)
print('')    

#finding accuracy for knn where k=5
test5=[]
predict5=[]

x=0
for i in range (folds*10):
    if(k5True_Table[i,0]==k5True_Table[i,1]):
        x=x+1
    test5.append(int(k5True_Table[i,0]))
    predict5.append(int(k5True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict5,test5)
print(cm)
print('')    

#finding accuracy for knn where k=6
test6=[]
predict6=[]

x=0
for i in range (folds*10):
    if(k6True_Table[i,0]==k6True_Table[i,1]):
        x=x+1
    test6.append(int(k6True_Table[i,0]))
    predict6.append(int(k6True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict6,test6)
print(cm)
print('')    

#finding accuracy for knn where k=7
test7=[]
predict7=[]

x=0
for i in range (folds*10):
    if(k7True_Table[i,0]==k7True_Table[i,1]):
        x=x+1
    test7.append(int(k7True_Table[i,0]))
    predict7.append(int(k7True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict7,test7)
print(cm)
print('')    

#finding accuracy for knn where k=8
test8=[]
predict8=[]

x=0
for i in range (folds*10):
    if(k8True_Table[i,0]==k8True_Table[i,1]):
        x=x+1
    test8.append(int(k8True_Table[i,0]))
    predict8.append(int(k8True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict8,test8)
print(cm)
print('')    

#finding accuracy for knn where k=9
test9=[]
predict9=[]

x=0
for i in range (folds*10):
    if(k9True_Table[i,0]==k9True_Table[i,1]):
        x=x+1
    test9.append(int(k9True_Table[i,0]))
    predict9.append(int(k9True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict9,test9)
print(cm)
print('')    

#finding accuracy for knn where k=10
test10=[]
predict10=[]

x=0
for i in range (folds*10):
    if(k10True_Table[i,0]==k10True_Table[i,1]):
        x=x+1
    test10.append(int(k10True_Table[i,0]))
    predict10.append(int(k10True_Table[i,1]))
accuracy_list=np.append(accuracy_list,(x/(folds*10))*100)
print('')


#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predict10,test10)
print(cm)
print('')    

#finding optimal k
optimalK=accuracy_list.argmax()
optimalK=optimalK+1
for k in range(1,11):
    print('k= ',k,' Accuracy= ',accuracy_list[k-1])


print('Optimal k= ',optimalK,' with accuracy= ',accuracy_list[optimalK] )

print('')
print('Applying optimal k to classify MNIST test data set')

#Applying optimal k to classify MNIST test data set


OptimalkTrue_Table = np.zeros((10000,2))

#finding nearest neighbor for optimal k
for i in range (10000):
    A=np.array(EuclideanDist[i])
    kneigbors=[]
    kneigbors=A.argsort()[:optimalK]
    kneighbor_label=[]
    for k in range(optimalK):
        kneighbor_label.append(train_label[kneigbors[k]])
    try:
        label=mode(kneighbor_label)
    except:
        label1=int(A.argmin())
        label=train_label[label1]            
    OptimalkTrue_Table[i,0]=test_label[i]
    OptimalkTrue_Table[i,1]=label

#finding accuracy for knn where k=Optimal k    
testoptimal=[]
predictoptimal=[]
for i in range (10000):
    testoptimal.append(int(OptimalkTrue_Table[i,0]))
    predictoptimal.append(int(OptimalkTrue_Table[i,1]))
    
x=0
for i in range (10000):
    if(OptimalkTrue_Table[i,0]==OptimalkTrue_Table[i,1]):
        x=x+1
accuracy=(x/10000)*100

print('knn accuracy for MNIST data set when k=',optimalK,'(optimal k): ',accuracy)

print('')

#rows are predicted and columns are actual
print("Rows are predicted labels and columns are actual labels")
print('')
cm=confusion_matrix(predictoptimal,testoptimal)
print(cm)
print('')    
