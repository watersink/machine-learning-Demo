# Machine Learning Demo
mlp,决策树,随机森林,bagging,adaboost,gbdt,xgboost,lightgbm,catboost,gcForest(deep forest),pca,svm

# 多层感知机mlp
    #python
    python3 mlp.py
    
    #cpp
    mkdir build
    cd build&&cmake ..
    make
    ./mlp

# 决策树decision_tree
    #python
    python3 decision_tree.py
    dot -Tpdf tree.dot -o tree.pdf
    
    #cpp
    mkdir build
    cd build&&cmake ..
    make
    ./decision_tree

# 随机森林random_forest
    #python
    python3 RandomForest.py
    
    #cpp
    mkdir build
    cd build&&cmake ..
    make
    ./random_forest

# 提升树boost_tree（bagging,adaboost,gbdt）
    #python
    python3 boost_tree.py
    
    #cpp
    mkdir build
    cd build&&cmake ..
    make
    ./boost_tree


# xgboost
    python3 xgDemo.py

# lightgbm
    python3 gbm.py

# catboost
    python3 cat.py

# gcForest(deep forest)
    python3 examples/wine.py
    python3 examples/demo_mnist.py


![img](https://github.com/watersink/machine-learning-Demo/blob/master/gcForest/gcForest.png)


# pca
    #降维
    python3 pca_Dimension_reduction.py
    #旋转数据
    python3 pca_rotate.py

# svm
    #python
    python3 svm.py
    
    #cpp
    mkdir build
    cd build&&cmake ..
    make
    ./svm
