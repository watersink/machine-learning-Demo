import xgboost
from numpy import loadtxt
from sklearn.svm import SVC,NuSVC,LinearSVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.datasets import load_wine,load_boston
from sklearn.utils import shuffle
import numpy as np


#分类
# 载入数据集
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,test_size=0.3)


model_svc = SVC(kernel='linear')
model_linearsvc = LinearSVC(loss='squared_hinge', multi_class='ovr')
model_nusvc = NuSVC(nu=0.5, kernel='rbf')

model_svc.fit(X_train, y_train)
model_linearsvc.fit(X_train, y_train)
model_nusvc.fit(X_train, y_train)

print("svc train score",model_svc.score(X_train, y_train))
print("linearsvc train score",model_linearsvc.score(X_train, y_train))
print("nusvc train score",model_nusvc.score(X_train, y_train))


# 保存模型
joblib.dump(model_svc, "svm.model") 
# 加载模型
#model_svc = joblib.load("svm.model")


#获得支持向量
print("支持向量: ",model_svc.support_vectors_)
#获得支持下向量的索引
print("支持下向量的索引: ",model_svc.support_)
#为每一个类别获得支持向量的数量
print("支持向量的数量: ",model_svc.n_support_)


# 对测试集做预测
y_pred = model_svc.predict(X_test)
#print(y_test,"    ",y_pred)
predictions = [round(value) for value in y_pred]
 
# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))





#回归
boston = load_boston()  # 加载波士顿房价回归数据集
X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例

svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_linear=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)

svr_rbf.fit(X_train,y_train)
svr_linear.fit(X_train,y_train)
svr_poly.fit(X_train,y_train)

print("svr_rbf train score",svr_rbf.score(X_train, y_train))
print("svr_linear train score",svr_linear.score(X_train, y_train))
print("svr_poly train score",svr_poly.score(X_train, y_train))

