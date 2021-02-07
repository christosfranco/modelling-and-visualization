import numpy
import linweighreg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# load data
data = pd.read_csv("insurance.csv", delimiter=",")
print("\n",data.head)
#Splitting into train and test sets
train_data, test_data = train_test_split(data,test_size=0.2)

train_data = numpy.array(train_data)

test_data = numpy.array(test_data)
#print(train_data.shape)
#print(test_data.shape)
X_train, y_train = (train_data[:,:-1]), (train_data[:,-1])
X_test, y_test = (test_data[:,:-1]), (test_data[:,-1])
#Splitting into features and results


# make sure that we have N-dimensional Numpy arrays (ndarray)
X_train = X_train.reshape((len(X_train), 1))
y_train = y_train.reshape((len(y_train), 1))
X_test = X_test.reshape((len(X_test), 1))
y_test = y_test.reshape((len(y_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (b) fit linear regression using only the first feature
model_single = linweighreg.LinearRegression()
model_single.fit(X_train[:,0], y_train)
wui = model_single.w
print("solve method\n" , wui)
model_single.fit2(X_train[:,0], y_train)
bui = model_single.w
print("inverse matrix method\n", bui)
print("Does both the methods yield the same result\n", numpy.allclose(wui,bui))

# (c) fit linear regression model using all features
model_alle = linweighreg.LinearRegression()
model_alle.fit(X_train, y_train)
sui = model_alle.w
print("Fit linear regression with all features\n" , sui)

# (d) evaluation of results
presingle = model_single.predict(X_test[:,0])
preall = model_alle.predict(X_test)
def rmse(t, tp):
    return numpy.sqrt(numpy.mean(numpy.square(numpy.abs(numpy.subtract(t,tp)))))

#This function is used to plot a linear line on the graph to compare visuali
def gtest(x):
    return x

#Values for the linear line
xtestvals = numpy.linspace(0,50,100) #100 points from 1 to 10 in ndarray
ytestvals = list(map(gtest, xtestvals)) #evaluate gtest for each point in xtestvals 

#The RMSE for both feature methods
print("RMSE Single Feature the Column 0:", rmse(y_test, (presingle)))
print("RMSE All features:", rmse(y_test, (preall)))

#plot with single feature
plt.scatter(y_test, presingle , alpha=0.5)
#plot with all features
plt.scatter(y_test, preall , alpha=0.5)
#linear test line 
plt.plot(xtestvals, ytestvals,alpha=0.5)

#Labels and title of plot
plt.xlabel("Test values")
plt.ylabel("Predictions")
plt.title("Evaluation, Orange is prediction with all features, Blue with single feature")
plt.show()
plt.close()
