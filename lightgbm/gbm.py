# coding: utf-8
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.datasets import load_wine,load_boston
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import label_binarize


#pip3 install lightgbm
#分类
# load or create your dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data,wine.target,test_size=0.3)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params={'learning_rate':0.1,
        'lambda_l1':0.1,
        'lambda_l2':0.2,
        'max_depth':6,
        'objective':'multiclass',
        'num_class':3,  
        }

print('Starting training...')
# train
gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('gbm.txt')
# load model
gbm = lgb.Booster(model_file='gbm.txt')

print('Starting predicting...')
# predict
#one hot output
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

predictions =[]
for value in y_pred:
    predictions.append(list(value).index(max(value)))
#print(predictions,"   ", y_test)
# eval
accuracy = accuracy_score(y_test, predictions)
print("test Accuracy: %.2f%%" % (accuracy * 100.0))


# 1、AUC
y_pred_pa = gbm.predict(X_test)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
y_test_oh = label_binarize(y_test, classes= [0,1,2])
print ('test auc：', roc_auc_score(y_test_oh, y_pred_pa, average='micro'))

#  2、混淆矩阵
y_pred = y_pred_pa .argmax(axis=1)
cm =confusion_matrix(y_test, y_pred )
print("confuse matrix: \n",cm)

#  3、经典-精确率、召回率、F1分数
precision_score(y_test, y_pred,average='micro')
recall_score(y_test, y_pred,average='micro')
f1=f1_score(y_test, y_pred,average='micro')
print("test f1 score:",f1)

# 4、模型报告
print(classification_report(y_test, y_pred))








#回归
# load or create your dataset
boston = load_boston()  # 加载波士顿房价回归数据集
X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }

print('Starting training...')
# train
gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('gbm.txt')
# load model
gbm = lgb.Booster(model_file='gbm.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
