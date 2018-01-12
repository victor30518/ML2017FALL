import os, sys
import numpy as np
from sklearn.decomposition import PCA

def load_data(test_data_path):    
    # 讀入測試資料
    x_test = []
    with open(test_data_path, 'r', encoding = 'utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            data = line.split(',')
            # 讀入句子
            x_test.append([int(data[1]),int(data[2].strip())])

    x_test = np.array(x_test)
    
    return x_test


img_set_path = sys.argv[1]
test_data_path = sys.argv[2]
predict_name = sys.argv[3]

image_set = np.load(img_set_path)

#輸入有多少成份我們想要留住分解
np.random.seed(666)
pca = PCA(n_components=400,whiten=True)
pca.fit(image_set)
x_pca = pca.transform(image_set)

# k-means
from sklearn import cluster, datasets, metrics
km = cluster.KMeans(n_clusters=2)  #K=2群
predict = km.fit_predict(x_pca)

# 讀test檔
x_test = load_data(test_data_path)

# 輸出predict
with open(predict_name,'w') as f:
    f.write('ID,Ans')
    f.write('\n')
    id = 0
    for case in x_test:
        if predict[case[0]] == predict[case[1]]:
            ans = 1
        else:
            ans = 0
        f.write(str(id)+',')
        f.write(str(ans))
        f.write('\n')
        id = id + 1