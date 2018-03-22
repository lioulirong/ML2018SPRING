import numpy as np 
import csv
import sys
import os
import errno
# read model
# scale_param = np.load('scale.npy')
args = sys.argv
w = np.load('tmp_model.npy')
offset = np.load('del_info.npy')

test_x = []
n_row = 0

text = open(args[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()

test_x = np.array(test_x)
# mean = scale_param[0]
# deviation = scale_param[1]
# test_x = (test_x-mean )/deviation

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


for i in range(len(offset)):
    off = offset[i]
    off -= i    
    test_x = np.delete(test_x,np.s_[1+off*9:1+off*9+9],1)

x_mean = np.mean(test_x)        
x_std = np.std(test_x)

for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if test_x[i][j] < 0:
            if j == 0 :
                test_x[i][j] = test_x[i][j+1]            
            else:
                test_x[i][j] = test_x[i][j-1]  
for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if abs(test_x[i][j]-x_mean) > 4*x_std:
            if j == 0 :
                test_x[i][j] = test_x[i][j+1]   
            else :
                test_x[i][j] = test_x[i][j-1]  

error = 0.0
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)
    

filename = args[2]
if not os.path.exists(os.path.dirname(filename)):
    try:
        if os.path.dirname(filename) == '':
            pass
        else : 
            os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

