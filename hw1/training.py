import csv 
import numpy as np
import random
import math
import sys

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

# select_feature = ("AMB_TEMP","NMHC","NO2","NOx","O3","RAINFALL","RH")
feature = (0,2,3,5,6,7,10,11,14)
# important_feature = ("PM10","PM2.5")
important_feature = (8,9)
x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        y.append(data[9][480*i+j+9])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            if t in important_feature:
                for s in range(9):
                    # 檢查是否有 -1 ,有則用上下各10個的平均值取代它
                    if data[t][480*i+j+s] == -1:
                        current = 480*i+j+s
                        data[t][current] = 0
                        for k in range(-10,11):
                            data[t][current] += data[t][current + k]
                        data[t][current] //= 21
                        print(data[t][current])
                    x[471*i+j].append(data[t][480*i+j+s])
                for s in range(5,9):
                    x[471*i+j].append(data[t][480*i+j+s]**2)
            if t in feature:
                for s in range(5,9):
                    x[471*i+j].append(data[t][480*i+j+s])

# 把y值是-1的訓練資料排除
iteration = len(y)
i = 0
while i < iteration:
    if y[i] < 0:
        del x[i]
        del y[i]
        iteration -= 1
    else:
        i += 1
        
x = np.array(x)
y = np.array(y)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 60000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
Lambda = 800

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss) + 2*Lambda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
np.save('model.npy',w)