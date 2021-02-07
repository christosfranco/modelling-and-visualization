import numpy
import sys
import linreg
import linweighreg
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
# Each row contains the following pieces of information:
# •CRIM - per capita crime rate by town
# •ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# •INDUS - proportion of non-retail business acres per town.
# •CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# •NOX - nitric oxides concentration (parts per 10 million)
# •RM - average number of rooms per dwelling
# •AGE - proportion of owner-occupied units built prior to 1940
# •DIS - weighted distances to five Boston employment centres
# •RAD - index of accessibility to radial highways
# •TAX - full-value property-tax rate per $10,000
# •PTRATIO - pupil-teacher ratio by town
# •B - 1000(Bk−0.63)**2 where Bk is the proportion of blacks by town
# •LSTAT - % lower status of the population
descrp = [' per capita crime rate by town','proportion of residential land zoned for lots over 25,000 sq.ft.',
' proportion of non-retail business acres per town.',' Charles River dummy variable (1 if tract bounds river; 0 otherwise)',' nitric oxides concentration (parts per 10 million)',' average number of rooms per dwelling',' proportion of owner-occupied units built prior to 1940',' weighted distances to five Boston employment centres',' index of accessibility to radial highways',' full-value property-tax rate per $10,000',' pupil-teacher ratio by town',' 1000(Bk−0.63)**2 where Bk is the proportion of blacks by town',' percentage lower status of the population']


###This is the house price we want to predict
# •MEDV - Median value of owner-occupied homes in $1000’s
####Write Desired feature to analyze dont choose CHAS as it 
# will produce a singular matrix. It has features with 0 or 1 
# not appropiate to analyze with linear regression 
####Set to None is all features are desired to be analyzed 
feature_to_analyse = 'B'

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")

def augment(X, max_order):
    """ Augments a given data
    matrix by adding additional 
    columns.
    
    NOTE: In case max_order is very large, 
    numerical inaccuracies might occur
    """
    
    X_augmented = X
    
    for i in range(2, max_order+1):
        X_augmented = numpy.concatenate([X_augmented, X**i], axis=1)
        
    return X_augmented

arr_features = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT',]

# maximum degree 
max_degree = 4

def save_feature_graph_cross_validation(feature_to_analyse):
    #get the feature index in dataset
    feature_in_int = arr_features.index(feature_to_analyse)
    X_train, y_train = train_data[:,feature_in_int], train_data[:,-1]
    X_test, y_test = test_data[:,feature_in_int], test_data[:,-1]
    print(X_train)
    # make sure that we have N-dimensional Numpy arrays (ndarray)
    X_train = X_train.reshape((len(X_train), 1))
    X_test = X_test.reshape((len(X_test), 1))
    y_train = y_train.reshape((len(y_train), 1))
    y_test = y_test.reshape((len(y_test), 1))
    print("Number of training instances: %i" % X_train.shape[0])
    print("Number of test instances: %i" % X_test.shape[0])
    #print("Number of features: %i" % X_train.shape[1])

    X_new_train = augment(X_train, max_degree)
    X_new_test  = augment(X_test, max_degree)
    # fit linear regression model using the augmented data matrix
    # TODO: Try out the following values: lam=0.0, lam=10**2, lam=10**5, lam=10**10, lam=10**15

    lam = 0
    if numpy.linalg.matrix_rank(X_new_train) < 2:
        pass
    model = linreg.LinearRegression(lam)
    model.fit(X_new_train,y_train)
    print("w values when lambda is set to 0:\n" , model.w)
    preds = model.predict(X_new_train)

    # same plot as before but with some more points
    # for plotting the model ...
    Xplot = numpy.arange(X_train.min(), X_train.max(), 0.01)
    Xplot = Xplot.reshape((len(Xplot), 1))
    Xplot = augment(Xplot, max_degree)
    preds_plot = model.predict(Xplot)

    fig = plt.figure()
    plt.plot(X_new_test[:,0], y_test, 'o', color='blue')
    plt.plot(X_new_train[:,0], preds, 'x', color='red')
    plt.plot(Xplot[:,0], preds_plot, '-', color='green')
    plt.xlabel(feature_to_analyse+descrp[feature_in_int], fontsize = 12)
    plt.ylabel('median value of owner-occupied homes in $1000', fontsize = 10)
    plt.show()
    fig.savefig(feature_to_analyse+'plot-linearregression.jpg')
    plt.close

    #Do leave one out cross validation
    #New lambda array
    lamarr = numpy.logspace(-8, 0, 100, base=10)
    lamarr = lamarr.reshape((len(lamarr),1))

    #arr for storing the error 
    errors_validation = []
    for lamval in lamarr:
        temp_error = 0

        model = linreg.LinearRegression(lam = lamval)
        for rows in range(0,26):
            #Take out some row of X_new_train and set X_val to 0 for that row the rest is train
            new_X_train = numpy.delete(X_new_train, rows, 0)
            X_val = [X_new_train[rows]]
            
            #Take out some of y_train row and set t_val to that row the rest is train
            new_y_train = numpy.delete(y_train, rows, 0)
            t_val = y_train[rows]
            
            #reshape
            #print("Shape of augmented data matrix: %s" % str(X_train_augmented.shape))
                
            # fit model on training set
            model.fit(new_X_train, new_y_train)

            # get validation predictions and error
            preds_val = model.predict(X_val)
            error_val = (t_val - preds_val)[0][0]**2
            temp_error += error_val
            
        #The leave one out cross validation error
        errors_validation.append(temp_error/len(X_new_train))  #temp_error

        #print("Training set: lam=%.10f and error_train=%.10f" % (lamarr[lamval], error_train))
        #print("Validation set: lam=%.10f and error_val=%.10f" % (lamval, temp_error/len(X_new_train)))

    #calculate the min error and give the index in the lamarr, return the lambda value
    bestlam = lamarr[errors_validation.index((min(errors_validation)))]
    print("the best lambda value is calculated to be: ", bestlam)


    # validation loss plot
    fig = plt.figure()
    plt.xticks(lamarr)
    # logscale for this plot since values are increasing rapidly!
    plt.xscale("log")
    plt.plot(lamarr, numpy.array(errors_validation))
    plt.xlabel('lambda', fontsize = 12)
    plt.ylabel('error', fontsize = 10)
    plt.show()
    fig.savefig(feature_to_analyse+'plot-lambda.jpg')


    ######################
    #Using the best lambda value to predict and plot the data
    model2 = linreg.LinearRegression(lam = bestlam)
    model2.fit(X_new_train,y_train)
    print("weights with the lambda value that gives the least error:\n" , model2.w)

    preds = model2.predict(X_new_test)

    # same plot as before but with some more points
    # for plotting the model ...
    Xplot = numpy.arange(X_test.min(), X_test.max(), 0.01)
    Xplot = Xplot.reshape((len(Xplot), 1))
    Xplot = augment(Xplot, max_degree)
    preds_plot = model2.predict(Xplot)

    fig = plt.figure()
    plt.plot(X_new_test[:,0], y_test, 'o', color='blue')
    plt.plot(X_new_test[:,0], preds, 'x', color='red')
    plt.plot(Xplot[:,0], preds_plot, '-', color='green')
    plt.xlabel(feature_to_analyse+descrp[feature_in_int], fontsize = 12)
    plt.ylabel('house price', fontsize = 10)
    plt.show()
    fig.savefig(feature_to_analyse+'plot-linearregression-bestlam.jpg')
    plt.close 
if feature_to_analyse != None:
    save_feature_graph_cross_validation(feature_to_analyse)
else:
    for i in arr_features:
        #Chas is a singular matrix with 0 and 1
        if i != 'CHAS':
            save_feature_graph_cross_validation(i)