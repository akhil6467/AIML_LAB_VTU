import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Read dataset to pandas dataframe
dataset = pd.read_csv("KNN-input.csv")
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(x.head())
Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10) 

classifier = KNeighborsClassifier(n_neighbors=1).fit(Xtrain, ytrain) 

ypred = classifier.predict(Xtest)

print ("\n-------------------------------------------------------------------------")
print ('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print ("-------------------------------------------------------------------------")
for label, pred in zip(ytest, ypred):
    print ('%-25s %-25s' % (label, pred), end="")
    if (label == pred):
        print (' %-25s' % ('Correct'))
    else:
        print (' %-25s' % ('Wrong'))
print ("-------------------------------------------------------------------------")
print('Accuracy of the classifer is %0.2f' % metrics.accuracy_score(ytest,ypred))
print ("-------------------------------------------------------------------------")

