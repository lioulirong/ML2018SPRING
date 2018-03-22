import csv
import numpy as np 
import math
from matplotlib import pyplot as plt 
"""
For tuning parameter 
Modified from sample code 
"""

file = open('train.csv','r',encoding = 'BIG5',newline = '')
rows = csv.reader(file,delimiter=',')

data = []
for i in range(18):
    data.append([])

#~row is more like a generator which dosen't store data
row_num = 0
for line in rows:
    if row_num != 0:
        for i in range(3,27):
            if line[i] == 'NR':
                data[row_num%18-1].append(float(0))
            else:
                data[row_num%18-1].append(float(line[i]))
    row_num += 1

file.close()
#parse x,y pairs
#we have 471*12 pairs
x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

#add bias
x = np.concatenate( (np.ones((x.shape[0],1)),x) , axis=1)


#split (x,y) pairs into
#-> train_x,train_y
#-> test_x,test_y
offset = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17];
for i in range(len(offset)):
    off = offset[i]
    off -= i    
    x = np.delete(x,np.s_[1+off*9:1+off*9+9],1)
#for line in x:
#    for element in line:
x_mean = np.mean(x)        
x_std = np.std(x)
#eliminate the negative 
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i][j] < 0:
            if j == 0:
                x[i][j] = x[i][j+1]    
            else :
                x[i][j] = x[i][j-1]
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if abs(x[i][j]-x_mean) > 4*x_std:
            if j == 0:
                x[i][j] = x[i][j+1]    
            else :
                x[i][j] = x[i][j-1]            

y_mean = np.mean(y)
y_std = np.std(y)
for i in range(len(y)):
    if y[i] < 0 :
        y[i] = y[i-1]
for i in range(len(y)):
    if abs(y[i]-y_mean) > 2*y_std:
        y[i] = y[i-1]

w = np.zeros(len(x[0]))
l_rate = 0.01
repeat = 1000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

loss_record = []

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    loss_record.append(cost_a)
#    print ('iteration: %d | Cost: %f ' % ( i,cost_a))

print ("finish training!")
#plt.plot(range(200),loss_record[0:200])
#plt.show()
np.save('del_info.npy',np.array(offset))
np.save('tmp_model.npy',w)





