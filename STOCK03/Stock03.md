# Stock
Stock Analyse System
# 股票预测系统详细设计

## 系统概述

随着中国经济社会的发展，更多的人选择投资股市来实现自己资产的升值，而股票的涨跌情况对于广大的股民来说，如何从上千只股票中选择优质股一直是一个难题。股票市场是我国证券业以及金融业不可缺少的组成部分，股票数据的分析与预测也具有重大的理论意义与实际意义。股票市场是一个极其复杂的动力学系统，高噪声、严重非线性和投资者的任意盲目性等因素决定了股票预测的复杂性。

本系统利用机器学习的方法，对输入的某股票进行过往数据的学习，然后将以往20天的股票数据作为输入，并且通过多种方法进行预测，从而预测出接下来股票的涨跌情况。

## 系统设计原理说明

### 1.分类模型选择

本系统分析比较了多种分类模型，包括决策树、SVM、KNN、逻辑回归和随机森林，实验结果证明其中随机森林模型的预测结果最好，故最终选择随机森林作为分类模型。

### 2.特征和标签设计

本系统的分析模型中，特征和标签设定如下：利用连续20个工作日的涨跌幅度预测第21天的涨跌幅度，因此把前20天的涨跌幅度作为一个20维的特征，然后根据第21天的涨跌幅度设定标签。实际观察得知股票在连续两个工作日内跌破5%或者涨过5%的情况较为少见，于是把跌破5%的情况设为标签-11，涨过5%的情况则设为标签11；-5%至+5%之间则每以0.5%作为区间，分20个标签，从-10到+10。这样就可以根据21天的涨跌幅度数据得到一组特征+标签。

### 3.模型有效性分析

为了衡量本系统的预测精确程度，从所有数据中，随机抽取90%作为训练集，剩下的10%作为测试集。用训练集训练完机器学习模型之后，再用测试集进行交叉验证，把预测结果和实际情况做对比，得到一个准确度，用于衡量模型的有效性。

## 系统框架设计

本系统使用网络爬虫技术，结合新浪的股票数据库采集数据集，然后通过sklearn框架进行数据的学习与预测。

### 系统框架图：

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/21.png)

### 数据流程图：

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/22.png)

### 用户操作流程及运行结果：

1.用户输入股票代码。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/23.png)

2.程序发送request请求到新浪股票数据库，采集从2000年开始到最近一个工作日的股票涨跌数据。如果这支股票在2000年以后才出现，则采集从发行日开始的数据。采集到的每天的数据都会输出到屏幕，格式为“日期+当日涨跌幅”。如图所示。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/24.png)

3.采集数据完成后，程序将历年股票涨跌情况绘制成图表，展示在屏幕上。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/25.png)

4.然后程序利用已有数据进行模型的训练，以及模型有效性分析，并把有效性分析的结果显示在屏幕上。以下图为例，测试集中共有441组数据，其中299组属于预测和实际相符的情况，计算得知总正确率为67.8%。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/26.png)

5.程序把最近20天的涨跌情况绘制成图表并显示。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/27.png)

6.根据最近20天的涨跌情况，训练好的模型预测明日股票涨跌情况。

![](https://github.com/chuochuoyouyugroup/Stock/blob/master/28.png)

## 不同预测方法比较：
### 1.决策树方法
![](https://raw.githubusercontent.com/YuBoyang0321151606/chuochuoyouyu/master/STOCK03/pic/%E5%86%B3%E7%AD%96%E6%A0%91.png)
### 2.SVM方法
![](https://raw.githubusercontent.com/YuBoyang0321151606/chuochuoyouyu/master/STOCK03/pic/SVM.png)
### 3.KNN方法
![](https://raw.githubusercontent.com/YuBoyang0321151606/chuochuoyouyu/master/STOCK03/pic/KNN.png)
### 4.LRC方法
![](https://raw.githubusercontent.com/YuBoyang0321151606/chuochuoyouyu/master/STOCK03/pic/LRC.png)
### 5.RFC方法
![](https://raw.githubusercontent.com/YuBoyang0321151606/chuochuoyouyu/master/STOCK03/pic/RFC.png)

由此可以看出决策树算法表现良好，因此我们采用决策树算法进行股票涨跌情况的预测。














