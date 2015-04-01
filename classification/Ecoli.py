__author__ = 'piyush'

from class_algos import Classification
from sklearn.feature_selection import VarianceThreshold

def featureReduction(X):
    sel = VarianceThreshold(threshold=(0.01*(1-0.01)))
    X_new = sel.fit_transform(X)
    if len(X_new[0])!=len(X[0]):
        print('Feature Vector Dimension Reduced by ' + str(len(X[0]) - len(X_new[0])))
    return X_new

def loadData(fileName):
    print("Loading Data...")

    data = []
    output = []
    for line in open(fileName):
        arr = line.split(',')
        X = arr[1:7]
        X = map(float, X)
        output.append(arr[8])
        data.append(X)
    print(len(data))
    data = featureReduction(data)
    return data, output

def ecoli():
    fileName = "/home/piyush/IdeaProjects/MachineLearning/ML_Ass/Classification/Ecoli/ecoli.data"
    X,Y = loadData(fileName)
    clf = Classification(X,Y)
    clf.supportvectormachine()
    clf.decisionTree()
    clf.fld()
    clf.logisticRegression()
    clf.NN()
    clf.NBC()

