# encoding=utf-8

import requests
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn import tree

print ("please input the stock code:")
stock_code = input()
stock_code = str(stock_code)
# 这是对一个股进行分析的代码
r = requests.get('http://q.stock.sohu.com/hisHq?'
                 '&code=cn_' + stock_code +
                 '&start=20000501'
                 '&end=20181010'
                 '&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp')
content = r.content
# 得到的json里有中文，需要解码:gbk->unicode(python默认编码格式)->utf-8
content = content.decode("gbk").encode('utf-8')
if bytes("non-existent", encoding = "utf8") in content:
    print ("The stock code you input does not exist!")
    input()
    exit()
pos_begin = content.find(bytes('[[', encoding = "utf8"))
pos_end = content.find(bytes(']]', encoding = "utf8"))
content = content[pos_begin+2:pos_end]
content = content.split(bytes('],[', encoding = "utf8"))
result = []
up_down_list = []
print ("[date  Change]")
for d in content:
    d = d[1:-1]
    d = d.split(bytes('","', encoding = "utf8"))
    for i_int in [7]:
        d[i_int] = int(d[i_int])
    for i_float in [1, 2, 3, 5, 6, 8]:
        d[i_float] = float(d[i_float])
    for i_f_percent in [4, 9]:
        d[i_f_percent] = float(d[i_f_percent][0:-1]) * 0.01
    result.append(d)
    up_down_list.append(d[4])
up_down_list.reverse()
result.reverse()
for d in result:
    print (d[0], " ", d[4]*100, "%")
print ("From", result[0][0], "to", result[-1][0], "totally", len(result), "days' stock history has been gathered, now analysing......")
for i in up_down_list:
    i = i * 1000
    i = round(i)
    i = int(i)
    i = i / 5
    if i != 0:
        if i > 0:
            i = i + 1
        else:
            i = i - 1
    if i > 10:
        i = 11
    if i < -10:
        i = -11
    labels_everyday.append(i)
# 画图展示一下所有天的涨跌幅
labels_everyday = labels_everyday[20:]
plt.title("Stock History Show")
plt.bar(range(len(labels_everyday)), labels_everyday)
plt.show()
# 构建数据集特征部分：X。前20天不可作为“结果“，故训练集规模-20。
features_all = []
for i in range(len(up_down_list)-20):
    X_item = []
    for j in range(20):
        X_item.append(up_down_list[i+j])
    features_all.append(X_item)
    
x = []
y = []
x_t = []
y_t = []

import random
train_No = random.sample(range(0, len(labels_everyday)), int(0.9 * len(labels_everyday)))


for i in range(0, len(features_all)):
    if i in train_No:
        x.append(features_all[i])
        y.append(labels_everyday[i])
    else:
        x_t.append(features_all[i])
        y_t.append(labels_everyday[i])
        
        
        
#决策树
clf = tree.DecisionTreeClassifier()
y = np.array(y, dtype = int)
clf.fit(x, y)
# 开始用测试集测定训练效果
predict_t = clf.predict(x_t)
total_count = 0
right_count = 0
for i in range(len(predict_t)):
    print ("predict:", predict_t[i], "  fact:", y_t[i])
    flag = predict_t[i] * y_t[i]
    total_count += 1
    if flag >= 0:
        right_count += 1
print ("---Accuracy Analyse Result---")
print ("Total:", total_count)
print ("Right:", right_count)
print ("Correct Rate:", right_count * 1.0 / total_count)

plt.title("Latest 20 Days")
plt.bar(range(len(up_down_list[len(up_down_list)-20:])), up_down_list[len(up_down_list)-20:])
plt.show()
tomorrow_predict_result = clf.predict([up_down_list[len(up_down_list)-20:]])
pre = tomorrow_predict_result[0] * 0.5
print ("Prediction: the Change tomorrow is likely to be（of decisiontree）:",)
if pre > 0:
    print ("+",)
print (pre, "%  ",)
print ("(Stock Code: ", stock_code, ")")




#SVM
from sklearn.svm import SVC   
clfSVM = SVC(kernel = 'linear')
y = np.array(y, dtype = int)
clfSVM.fit(x, y)

predict_t = clfSVM.predict(x_t)
total_count = 0
right_count = 0
for i in range(len(predict_t)):
    print ("predict:", predict_t[i], "  fact:", y_t[i])
    flag = predict_t[i] * y_t[i]
    total_count += 1
    if flag >= 0:
        right_count += 1
print ("---Accuracy Analyse Result---")
print ("Total:", total_count)
print ("Right:", right_count)
print ("Correct Rate:", right_count * 1.0 / total_count)

plt.title("Latest 20 Days")
plt.bar(range(len(up_down_list[len(up_down_list)-20:])), up_down_list[len(up_down_list)-20:])
plt.show()
tomorrow_predict_result = clfSVM.predict([up_down_list[len(up_down_list)-20:]])
pre = tomorrow_predict_result[0] * 0.5
print ("Prediction: the Change tomorrow is likely to be（of SVM）:",)
if pre > 0:
    print ("+",)
print (pre, "%  ",)
print ("(Stock Code: ", stock_code, ")")






#KNN
from sklearn.neighbors import KNeighborsClassifier
clfKNN = KNeighborsClassifier()
y = np.array(y, dtype = int)
clfKNN.fit(x, y)
predict_t = clfKNN.predict(x_t)
total_count = 0
right_count = 0
for i in range(len(predict_t)):
    print ("predict:", predict_t[i], "  fact:", y_t[i])
    flag = predict_t[i] * y_t[i]
    total_count += 1
    if flag >= 0:
        right_count += 1
print ("---Accuracy Analyse Result---")
print ("Total:", total_count)
print ("Right:", right_count)
print ("Correct Rate:", right_count * 1.0 / total_count)

plt.title("Latest 20 Days")
plt.bar(range(len(up_down_list[len(up_down_list)-20:])), up_down_list[len(up_down_list)-20:])
plt.show()
tomorrow_predict_result = clfKNN.predict([up_down_list[len(up_down_list)-20:]])
pre = tomorrow_predict_result[0] * 0.5
print ("Prediction: the Change tomorrow is likely to be（of KNN）:",)
if pre > 0:
    print ("+",)
print (pre, "%  ",)
print ("(Stock Code: ", stock_code, ")")



#Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
clfLRC = LogisticRegression(penalty='l2')
y = np.array(y, dtype = int)
clfLRC.fit(x, y)
predict_t = clfLRC.predict(x_t)
total_count = 0
right_count = 0
for i in range(len(predict_t)):
    print ("predict:", predict_t[i], "  fact:", y_t[i])
    flag = predict_t[i] * y_t[i]
    total_count += 1
    if flag >= 0:
        right_count += 1
print ("---Accuracy Analyse Result---")
print ("Total:", total_count)
print ("Right:", right_count)
print ("Correct Rate:", right_count * 1.0 / total_count)

plt.title("Latest 20 Days")
plt.bar(range(len(up_down_list[len(up_down_list)-20:])), up_down_list[len(up_down_list)-20:])
plt.show()
tomorrow_predict_result = clfLRC.predict([up_down_list[len(up_down_list)-20:]])
pre = tomorrow_predict_result[0] * 0.5
print ("Prediction: the Change tomorrow is likely to be（of LRC）:",)
if pre > 0:
    print ("+",)
print (pre, "%  ",)
print ("(Stock Code: ", stock_code, ")")





#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clfRFC = RandomForestClassifier(n_estimators=8)
y = np.array(y, dtype = int)
clfRFC.fit(x, y)
predict_t = clfRFC.predict(x_t)
total_count = 0
right_count = 0
for i in range(len(predict_t)):
    print ("predict:", predict_t[i], "  fact:", y_t[i])
    flag = predict_t[i] * y_t[i]
    total_count += 1
    if flag >= 0:
        right_count += 1
print ("---Accuracy Analyse Result---")
print ("Total:", total_count)
print ("Right:", right_count)
print ("Correct Rate:", right_count * 1.0 / total_count)

plt.title("Latest 20 Days")
plt.bar(range(len(up_down_list[len(up_down_list)-20:])), up_down_list[len(up_down_list)-20:])
plt.show()
tomorrow_predict_result = clfRFC.predict([up_down_list[len(up_down_list)-20:]])
pre = tomorrow_predict_result[0] * 0.5
print ("Prediction: the Change tomorrow is likely to be（of RFC）:",)
if pre > 0:
    print ("+",)
print (pre, "%  ",)
print ("(Stock Code: ", stock_code, ")")
input()
