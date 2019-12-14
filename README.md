# Machine Learning Demo
决策树,xgboost,pca

# 决策树
    #python
    python3 decision_tree.py
    dot -Tpdf tree.dot -o tree.pdf
    
    #cpp
    mkdir build
    cmake ..
    make
    ./decision_tree ../air.csv
    
# xgboost
    python3 xgDemo.py

# pca
    #降维
    python3 pca_Dimension_reduction.py
    #旋转数据
    python3 pca_rotate.py



