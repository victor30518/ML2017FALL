import csv
import numpy as np
import random
import math
import sys

# read model
w = np.load('model.npy')

test_x = []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text , delimiter= ",")

select_feature = ("AMB_TEMP","CO","NMHC","NO2","NOx","O3","RAINFALL","RH","WD_HR")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(7,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        if r[1] == "PM10" or r[1] == "PM2.5":
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]))
            for i in range(7,11):
                test_x[n_row//18].append(float(r[i])**2)
        if r[1] in select_feature:
            for i in range(7,11):
                if r[i] !="NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()