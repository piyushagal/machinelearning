__author__ = 'piyush'

from class_algos import Classification
from sklearn.feature_selection import VarianceThreshold

def featureReduction(X):
    sel = VarianceThreshold(threshold=(0.8*(1-0.8)))
    X_new = sel.fit_transform(X)
    if len(X_new[0])!=len(X[0]):
        print('Feature Vector Dimension Reduced by ' + str(len(X_new[0])-len(X[0])))
    return X_new

def loadData(fileName):
    print("Loading Data...")

    data = []
    output = []
    for line in open(fileName):
        if(line.__contains__('Study')):
            continue
        arr = line.split(',')
        X = arr[3:21]
        X = map(float, X)
        output.append(arr[21][0])
        data.append(X)
    output = map(int, output)
    return data, output

def climateModel():
    fileName = "/home/piyush/IdeaProjects/MachineLearning/ML_Ass/Classification/climateModel/pop_failures.dat"
    X,Y = loadData(fileName)
    clf = Classification(X,Y)
    clf.supportvectormachine()
    clf.decisionTree()
    clf.fld()
    clf.logisticRegression()
    clf.NN()
    clf.NBC()

