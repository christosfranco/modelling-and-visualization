import numpy
import linreg
import linweighreg
import matplotlib.pyplot as plt 

# maximum degree 
max_degree = 4
# load data
train_data = numpy.loadtxt("men-olympics-100.txt", delimiter=" ")


X_train, Y_train = train_data[:,0], train_data[:,1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
X_train = X_train.reshape((len(X_train), 1))
Y_train = Y_train.reshape((len(Y_train), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of features: %i" % X_train.shape[1])
print(X_train)
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

Xnew = augment(X_train, max_degree)
# fit linear regression model using the augmented data matrix
# TODO: Try out the following values: lam=0.0, lam=10**2, lam=10**5, lam=10**10, lam=10**15

lam = 0
model = linreg.LinearRegression(lam)
model.fit(Xnew,Y_train)
print("w values when lambda is set to 0:\n" , model.w)
preds = model.predict(Xnew)

# same plot as before but with some more points
# for plotting the model ...
Xplot = numpy.arange(X_train.min(), X_train.max(), 0.01)
Xplot = Xplot.reshape((len(Xplot), 1))
Xplot = augment(Xplot, max_degree)
preds_plot = model.predict(Xplot)

fig = plt.figure()
plt.plot(Xnew[:,0], Y_train, 'o')
plt.plot(Xnew[:,0], preds, 'x', color='red')
plt.plot(Xplot[:,0], preds_plot, '-', color='red')
plt.xlabel('years', fontsize = 12)
plt.ylabel('time', fontsize = 10)
plt.show()
fig.savefig('plot-linearregression.jpg')
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
        #Take out some row of Xnew and set X_val to 0 for that row the rest is train
        new_X_train = numpy.delete(Xnew, rows, 0)
        X_val = [Xnew[rows]]
        
        #Take out some of Y_train row and set t_val to that row the rest is train
        new_Y_train = numpy.delete(Y_train, rows, 0)
        t_val = Y_train[rows]
        
        #reshape
        #print("Shape of augmented data matrix: %s" % str(X_train_augmented.shape))
            
        # fit model on training set
        model.fit(new_X_train, new_Y_train)

        # get validation predictions and error
        preds_val = model.predict(X_val)
        error_val = (t_val - preds_val)[0][0]**2
        temp_error += error_val
        
    #The leave one out cross validation error
    errors_validation.append(temp_error/len(Xnew))  #temp_error

    #print("Training set: lam=%.10f and error_train=%.10f" % (lamarr[lamval], error_train))
    print("Validation set: lam=%.10f and error_val=%.10f" % (lamval, temp_error/len(Xnew)))

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
fig.savefig('plot-lambda.jpg')


######################
#Using the best lambda value to predict and plot the data
model2 = linreg.LinearRegression(lam = bestlam)
model2.fit(Xnew,Y_train)
print("weights with the lambda value that gives the least error:\n" , model2.w)

preds = model2.predict(Xnew)

# same plot as before but with some more points
# for plotting the model ...
Xplot = numpy.arange(X_train.min(), X_train.max(), 0.01)
Xplot = Xplot.reshape((len(Xplot), 1))
Xplot = augment(Xplot, max_degree)
preds_plot = model2.predict(Xplot)

fig = plt.figure()
plt.plot(Xnew[:,0], Y_train, 'o')
plt.plot(Xnew[:,0], preds, 'x', color='red')
plt.plot(Xplot[:,0], preds_plot, '-', color='red')
plt.xlabel('years', fontsize = 12)
plt.ylabel('time', fontsize = 10)
plt.show()
fig.savefig('plot-linearregression-bestlam.jpg')
plt.close 