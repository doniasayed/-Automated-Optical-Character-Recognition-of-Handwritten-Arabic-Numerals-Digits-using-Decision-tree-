import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

class Decisiontree:
#using pandas to read the dataset
  x_train = pd.read_csv("csvTrainImages 60k x 784.csv")
  y_train = pd.read_csv("csvTrainLabel 60k x 1.csv")

#Create the Decision tree classifier and fit it to the the train images
  clf = DecisionTreeClassifier()
  clf.fit(x_train, y_train)

  x_test = pd.read_csv("csvTestImages 10k x 784.csv")
  y_test = pd.read_csv("csvTestLabel 10k x 1.csv")

  y_pred = clf.predict(x_test)
#print the accuracy of the decision tree model
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#loop to predict the number from the testimages
  x=0
  while os.path.isfile("csvTestImages 10k x 784.csv"):
    #iloc() function enables us to select a particular cell of the dataset
    y_predicted=clf.predict((x_test.iloc[x].values).reshape(1,-1))
    pixel=x_test.iloc[x]
    y_pred=clf.predict(x_test)
    acc=metrics.accuracy_score(y_pred,y_test)
    pixel=np.array(pixel,dtype='uint8')
    pixel=pixel.reshape((28,28))
    plt.title('its probabily = {y_predicted} , Accuracy = {acc}'.format(y_predicted=y_predicted,acc=acc))
    pixel=np.transpose(pixel)
    plt.imshow(pixel,cmap='gray')
    plt.show()
    x+=1
    





        