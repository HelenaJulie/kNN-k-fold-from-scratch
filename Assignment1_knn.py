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
