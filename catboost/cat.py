from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt 
#pip3 install catboost

#分类
wine = load_wine()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
print(wine.items)
categorical_features_indices = np.where(Xtrain.dtype != np.float)[0]
#catboost = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices ,learning_rate=0.5, loss_function='Logloss',logging_level='Verbose')#2分类`
catboost = CatBoostClassifier(iterations=10, depth=5,cat_features=categorical_features_indices ,learning_rate=0.5, loss_function='MultiClass',logging_level='Verbose')#多分类
catboost.fit(Xtrain,Ytrain)
score_r = catboost.score(Xtest,Ytest)

print("catboost:{}".format(score_r))
# 保存模型
joblib.dump(catboost, "catboost.model") 
# 加载模型
catboost = joblib.load("catboost.model")


#交叉验证
catboost_score = cross_val_score(catboost,wine.data,wine.target,cv=10).mean()
print("10 folder val score: ",catboost_score)


# 对测试集做预测
y_pred = catboost.predict(Xtest)
predictions = [np.round(value) for value in y_pred]
 
# 评估预测结果
accuracy = accuracy_score(Ytest, predictions)
print("test Accuracy: %.2f%%" % (accuracy * 100.0))

#特征重要性
fea_ = catboost.feature_importances_
fea_name = catboost.feature_names_
plt.figure(figsize=(10, 10))
plt.barh(fea_name,fea_,height =0.5)
plt.show()



#回归
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
boston = load_boston()  # 加载波士顿房价回归数据集
X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例
cbr = CatBoostRegressor(iterations=10,depth=5,learning_rate=0.5,loss_function='RMSE')
cbr.fit(X1, y1)
print('catboost regress mse: ',mean_squared_error(y_test, cbr.predict(X_test)))
print('feature_importances :',cbr.feature_importances_)

