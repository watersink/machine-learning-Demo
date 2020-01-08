
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine
from sklearn.externals import joblib
import numpy as np


wine = load_wine()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)



#bagging
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
bagging.fit(Xtrain,Ytrain)
score_r = bagging.score(Xtest,Ytest)

print("bagging score:{}".format(score_r))
# 保存模型
joblib.dump(bagging, "bagging.model") 
# 加载模型
bagging = joblib.load("bagging.model")



bagging_score = cross_val_score(bagging,wine.data,wine.target,cv=10).mean()
print("10 folder val score: ",bagging_score)





#adaboost
adaboost = AdaBoostClassifier(random_state=0,n_estimators=25)
adaboost.fit(Xtrain,Ytrain)
score_r = adaboost.score(Xtest,Ytest)

print("adaboost score:{}".format(score_r))
# 保存模型
joblib.dump(adaboost, "adaboost.model") 
# 加载模型
adaboost = joblib.load("adaboost.model")



adaboost_score = cross_val_score(adaboost,wine.data,wine.target,cv=10).mean()
print("10 folder val score: ",adaboost_score)





#gbdt classify
gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
gbdt.fit(Xtrain,Ytrain)
score_r = gbdt.score(Xtest,Ytest)

print("gbdt score:{}".format(score_r))
# 保存模型
joblib.dump(gbdt, "gbdt.model") 
# 加载模型
gbdt = joblib.load("gbdt.model")



gbdt_score = cross_val_score(gbdt,wine.data,wine.target,cv=10).mean()
print("10 folder val score: ",gbdt_score)




#gbdt regress
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
boston = load_boston()  # 加载波士顿房价回归数据集
X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例
clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')
clf.fit(X1, y1)
print('GDBT regress mse: ',mean_squared_error(y_test, clf.predict(X_test)))
# print('每次训练的得分记录：',clf.train_score_)
print('feature_importances :',clf.feature_importances_)
plt.plot(np.arange(500), clf.train_score_, 'b-')  # 绘制随着训练次数增加，训练得分的变化
plt.show()



