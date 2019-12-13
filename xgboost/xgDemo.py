import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np

# 载入数据集
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y

X = dataset[:,0:8]
Y = dataset[:,8]
 
# 把数据集拆分成训练集和测试集
seed = 7
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



#X_train=(X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
#X_test=(X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))



# 拟合XGBoost模型
# 多分类
#model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=50, silent=True, objective='multi:softmax')
# 二分类
model = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=50, silent=True, objective='binary:logistic')
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "xg.model") 
# 加载模型
model = joblib.load("xg.model")
# 对测试集做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
 
# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
