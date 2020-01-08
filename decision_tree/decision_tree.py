#！／user/bin/env python
#-*- coding:utf-8 -*-
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import pydot
import numpy as np

#dot -Tpdf tree.dot -o tree.pdf


with open("air.csv","r",encoding="utf-8") as f:
    lines=f.readlines()

label=[]
data=[]
for line in lines:
    data.append(line.rstrip("\n").split(",")[:-1])
    label.append(line.rstrip("\n").split(",")[-1])

data=np.asarray(data)
label=np.asarray(label)
print(data.shape)
print(label.shape)



#参数random_state是指随机生成器，0表示函数输出是固定不变的
X_train,X_test,y_train,y_test = train_test_split(data, label,random_state=42,test_size=0.2)

for depth in list(range(1,16)):
    tree = DecisionTreeClassifier(max_depth=depth,random_state=0)
    tree.fit(X_train,y_train)
    print('tree depth:{}   Train score:{:.3f}    Test score:{:.3f}'.format(depth,tree.score(X_train,y_train),tree.score(X_test,y_test)))



# 保存模型
joblib.dump(tree, "tree.model") 
# 加载模型
tree = joblib.load("tree.model")

print (classification_report(tree.predict(X_test),y_test,target_names=['score1','score2','score3']))

#生成可视化图
export_graphviz(tree,out_file="tree.dot",class_names=['score1','score2','score3'],feature_names=['wind','snow','rain','sunny'],impurity=False,filled=True)
#展示可视化图
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('./tree.png')
