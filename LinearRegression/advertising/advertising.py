import numpy
import linweighreg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



###Choose Single Feature Column
#singleFeatureColumn = 0
singleFeatureColumn = list(range(0,3))

# load data
data = pd.read_csv("Advertising.csv", delimiter=",", index_col=False)
print("\n",data.head)

#dropping unnamed cloumn
data = data.drop(['Unnamed: 0'], axis=1)

#print(data.head)

#Splitting into train and test sets
train_data, test_data = train_test_split(data,test_size=0.4)

#print(train_data.shape)
#print(test_data.shape)

#Splitting into features and results
X_train = train_data.drop('sales',axis=1)
y_train = train_data['sales']

X_test = test_data.drop('sales',axis=1)
y_test = test_data['sales']

X_train = pd.DataFrame.to_numpy(X_train)
y_train = pd.DataFrame.to_numpy(y_train)
X_test = pd.DataFrame.to_numpy(X_test)
y_test = pd.DataFrame.to_numpy(y_test)


#Print info on data set
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# fit linear regression using only the first feature
#using the solve method
singleFitArr = []
for i in singleFeatureColumn:
    model_single1 = linweighreg.LinearRegression()
    model_single1.fit(X_train[:,i], y_train)
    wui = model_single1.w
    print("solve method\n" , wui)
    #inverse matrix method
    model_single = linweighreg.LinearRegression()
    model_single.fit2(X_train[:,i], y_train)
    bui = model_single.w
    print("inverse matrix method\n", bui)
    print("Does both the methods yield the same result\n", numpy.allclose(wui,bui))
    predsingle = model_single.predict(X_test[:,i])
    singleFitArr.append(predsingle)

# fit linear regression model using all features
model_alle = linweighreg.LinearRegression()
model_alle.fit2(X_train, y_train)
sui = model_alle.w
print("Fit linear regression with all features\n" , sui)

# evaluation of results
preall = model_alle.predict(X_test)

#Finding the Root Mean Square Error between the test values and the predicted values
#Can use another error function that makes great outliers weighting more, to ensure that we do not get a too wide destribution on the predictions.
def rmse(y_true, y_pred):
    return numpy.sqrt(numpy.mean(numpy.square(numpy.abs(numpy.subtract(y_true,y_pred)))))

def mape(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


#This function is used to plot a linear line on the graph to compare visuali
def gtest(x):
    return x

#Values for the linear line
xlinvals = numpy.linspace(min(y_test),max(y_test),100) #100 points from 1 to 10 in ndarray
ylinvals = list(map(gtest, xlinvals)) #evaluate gtest for each point in xlinvals 

#The RMSE and MAPE
for i in range(len(singleFeatureColumn)):
    print("RMSE Single Feature the Column" ,singleFeatureColumn[i] ," :" , rmse(y_test, (singleFitArr[i])))
    print("MAPE Single Feature the Column" ,singleFeatureColumn[i] ," :" , mape(y_test, (singleFitArr[i])))
    #plot with single feature
    plt.scatter(y_test, singleFitArr[i] , alpha=0.5)

print("RMSE All features:", rmse(y_test, (preall)))
print("MAPE All features:", mape(y_test, (preall)))


#plot with all features
plt.scatter(y_test, preall , alpha=0.5)
#linear test line 
plt.plot(xlinvals, ylinvals,alpha=0.5)

#plot with single feature
#plt.scatter(y_test, presingle , alpha=0.5)
#plot with all features
#plt.scatter(y_test, preall , alpha=0.5)
#linear test line 
#plt.plot(xlinvals, ylinvals,alpha=0.5)

#Labels and title of plot
plt.xlabel("Test values")
plt.ylabel("Predictions")
title = ""
colors = ["blue", "orange", "green", "red","cyan","magenta","yellow", "black" ,"white"]
for i in range(len(singleFeatureColumn)):
    title = title + ("Column " +  str(i) + " is: " + colors[i]+ "   ")

title = title + "all features are " + colors[len(singleFeatureColumn)]

plt.title(title)
plt.show()
plt.close()


#If we use all features we get a higher RMSE and MAPE. 
# But our prediction values are better at foreing a wider range 