__author__ = 'piyush'

from class_algos import Classification
from sklearn.feature_selection import VarianceThreshold

def featureReduction(X):
    sel = VarianceThreshold(threshold=(0.8*(1-0.8)))
    X_new = sel.fit_transform(X)
    return X_new

def loadData(fileName):
    print("Loading Data...")

    data = []
    output = []
    for line in open(fileName):
        line = line.replace('?', '0')
        arr = line.split(",")
        X = arr[1:10]
        X = map(int, X)
        output.append(arr[10][0])
        data.append(X)
    data = featureReduction(data)
    output = map(int, output)
    return data, output

def breastCancer():
    fileName = "/home/piyush/IdeaProjects/MachineLearning/ML_Ass/Classification/Breast_Cancer/breast-cancer-wisconsin.data"
    X,Y = loadData(fileName)
    clf = Classification(X,Y)
    clf.supportvectormachine()
    clf.decisionTree()
    clf.fld()
    clf.logisticRegression()
    clf.NN()
    clf.NBC()
