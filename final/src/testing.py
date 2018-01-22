import io
import os, sys
import csv
import jieba
import math
import numpy as np
import sklearn.metrics.pairwise as sk
from gensim.models import word2vec

def parse_data(data):
    return [ (sent.replace('A:','')).replace('B:','') for sent in data]

def jieba_seg(lines):
    seg_line = []
    words = jieba.cut(lines)
    for word in words:
        if word != ' ' and word != '':
            seg_line.append(word)
    return seg_line

def vector_sim(quest,opt,model):
    max_sim = 0
    max_sim2 = 0
    opt_len = len(opt)
    
    quest_seg = []
    opt_seg = []
    quest_avg_vector = np.zeros(wv_dim)
    
    num = 0
    for q in quest:
        q_seg = jieba_seg(q)

        # 算平均向量
        for i in q_seg:
            try:
                quest_avg_vector += model.wv[i]
                num = num + 1
            except Exception as e:
                pass
    if num != 0:
        quest_avg_vector /= num
        
    for o in opt:
        o_seg = jieba_seg(o)
        if o_seg is None:
            continue
        opt_seg.append(o_seg)
    
    sum_sim = np.zeros((opt_len,1),dtype=float)

    for i,one_opt in enumerate(opt_seg):
        count_sim_time = 0
        num = 0
        # 計算其中一個選項的平均向量
        one_opt_avg_vector = np.zeros(wv_dim)
        for opt_seg in one_opt:
            try:
                one_opt_avg_vector += model.wv[opt_seg]
                num = num + 1
            except Exception as e:
                pass
        if num != 0:
            one_opt_avg_vector /= num

        sum_sim[i] = sk.cosine_similarity([quest_avg_vector],[one_opt_avg_vector])

    return sum_sim

def check_sim(data_file,model):
    index=[]
    questions=[]
    options=[]
    answer=[]
    model_output=[]
    model_output2=[]
    
    with io.open(data_file,'r',encoding='utf-8') as content:
        for line in content:
            lines = line.split(',')
            if len(lines) == 3:
                index.append(lines[0])
                dialog_ = lines[1].split('\t')
                questions.append(parse_data(dialog_))
                options_ = lines[2].split('\t')
                options.append(parse_data(options_))
                try:
                    answer.append(int(lines[3][:-1]))
                except:
                    pass
            
        del index[0]
        del questions[0]
        del options[0]
        
    M = []
    for i,question in enumerate(questions):
        mat = vector_sim(question,options[i],model)
        M.append(mat)
    return M

# 參數設定
wv_dim = 64
jieba.set_dictionary('../dict_TW.txt')
test_data_path = sys.argv[1]

# 設定用來預測的 model
model1_path = "../model1.w2v"
model2_path = "../model2.w2v"
model3_path = "../model3.w2v"
model4_path = "../model4.w2v"

# predict
model = word2vec.Word2Vec.load(model1_path)
predict_1 = check_sim(test_data_path,model)
# print("predict_1 結束")

model = word2vec.Word2Vec.load(model2_path)
predict_2 = check_sim(test_data_path,model)
# print("predict_2 結束")

model = word2vec.Word2Vec.load(model3_path)
predict_3 = check_sim(test_data_path,model)
# print("predict_3 結束")

model = word2vec.Word2Vec.load(model4_path)
predict_4 = check_sim(test_data_path,model)
# print("predict_4 結束")

# 合併
predict = np.array(predict_1) + np.array(predict_2) + np.array(predict_3) + np.array(predict_4)

# 選擇最大相似度的選項作為答案
predict = np.argmax(predict,axis=1)
predict = predict.squeeze()

# 產生CSV
filename = "predict/ensemble_4.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","ans"])
for i in range(len(predict)):
    s.writerow([str(i+1),predict[i]]) 
text.close()