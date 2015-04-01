__author__ = 'piyush'


from sklearn import svm, tree, lda, neighbors, naive_bayes, cross_validation
from sklearn.linear_model import LogisticRegression

class Classification():

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def getAccuracy(self,predicted, actual):
        count = 0
        for i in range(0, len(predicted)):
            if predicted[i] == actual[i]:
                count+=1
        return float(count)/len(predicted)

    def supportvectormachine(self):
        print('Classification Using Support Vector Machine ....')
        clf = svm.LinearSVC()
        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Support vector Machine is : ' + str(accuracy*100) + ' %')


    def decisionTree(self):
        print('Classification Using Decision Tree ....')
        clf = tree.DecisionTreeClassifier()
        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Decision Tree is : ' + str(accuracy*100) + ' %')

    def fld(self):
        print('Classification Using Fisher Linear Discriminant ....')

        clf = lda.LDA()

        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit_transform(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Linear Discriminant is : ' + str(accuracy*100) + ' %')

    def logisticRegression(self):
        print('Classification Using Logistic Regression ....')

        clf = LogisticRegression(penalty='l2', tol=.01)
        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Logistic Regression is : ' + str(accuracy*100) + ' %')

    def NN(self):
        print('Classification Using Nearest Neighbours ....')

        clf = neighbors.KNeighborsClassifier(5, weights='distance')

        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Nearest Neighbours is : ' + str(accuracy*100) + ' %')

    def NBC(self):
        print('Classification Using Naive Bayes Classifier ....')
        clf = naive_bayes.GaussianNB()
        X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.40, random_state=0)
        X_test, X_validate,Y_test, Y_validate = cross_validation.train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

        clf.fit(X_train, Y_train)

        predicted = clf.predict(X_test)
        accuracy = self.getAccuracy(predicted, Y_test)
        print('Accuracy of Naive Bayes is : ' + str(accuracy*100) + ' %')