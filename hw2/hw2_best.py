import numpy as np
import csv
import sys
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

# 設定檔案路徑名稱
X_train_filename = sys.argv[1]
Y_train_filename = sys.argv[2]
X_test_filename = sys.argv[3]

# 讀入DATASET
X_train = np.loadtxt(X_train_filename, delimiter=",",skiprows=(1))
Y_train = np.loadtxt(Y_train_filename, delimiter=",",skiprows=(1))
X_test = np.loadtxt(X_test_filename, delimiter=",",skiprows=(1))

# AdaBoost
abc =  AdaBoostClassifier()
# 10-fold cross validation
score = cross_val_score(abc,X_train,Y_train,cv=10,scoring='accuracy')
print(score.mean())

# 用所有資料來建模
abc.fit(X_train,Y_train)
a = abc.predict(X_test)

# 輸出CSV
ans = []
for i in range(len(X_test)):
    ans.append([str(i+1)])
    ans[i].append(str(int(a[i])))

filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()