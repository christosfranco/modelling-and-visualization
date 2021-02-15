import csv
import numpy as np
import matplotlib.pyplot as plt

import mhsampler as mh


def readdataset(filename):
    """readdataset - reads the sunspot data set from the file with filename.
       Returns tuple of (X, t)."""
    X_str = []
    t_str = []
    with open(filename, newline='') as datafile:
        data_reader = csv.reader(datafile, delimiter=' ')
        for row in data_reader:
            X_str.append(row[0:5])
            t_str.append(row[5])

    X = np.array(X_str, dtype=np.float)
    t = np.array(t_str, dtype=np.float)
    return X, t


# Read the training set
X_train, t_train = readdataset('../../data/sunspotsTrainStatML.dt')

N_train, D = X_train.shape
print("Training set has X dimension D = " + str(D) + " and N = " + str(N_train) + ' samples.')


# Read the test set
X_test, t_test = readdataset('../../data/sunspotsTestStatML.dt')

N_test, D_test = X_test.shape
print("Test set has X dimension D = " + str(D_test) + " and N = " + str(N_test) + ' samples.')


# Visualize the data set
plt.figure()
plt.plot(X_train[:,4], t_train, 'o')
plt.title("Train")
plt.xlabel('X[4]')
plt.ylabel('t')

plt.figure()
plt.plot(X_test[:,4], t_test, 'o')
plt.title("Test")
plt.xlabel('X[4]')
plt.ylabel('t')

plt.figure()
plt.hist(t_train)
plt.title('Train')
plt.xlabel('t values')
plt.ylabel('hist(t)')

plt.figure()
plt.hist(t_test)
plt.title('Test')
plt.xlabel('t values')
plt.ylabel('hist(t)')



# Experimenting
# #w = np.random.rand(6)
# w = np.zeros(6)
# w[0] = 10.0
# sigma_square = 1.0
# mu_prior = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float)
#
# print("w = " + str(w))
# print("Prior = " + str(mh.prior(w, mu_prior, sigma_square)))
# print("likelihood = " + str(mh.likelihood(w, t_train, X_train)))
# print("Unormalized posterior = " + str(mh.posterior_unorm(w, t_train, X_train, mu_prior, sigma_square)))


# Make predictions based on the model
N_samples = 1000
sigma_square = 0.5**2

# Selection 1 (features from t-1)
mu_prior = np.array([1.0, 1.0], dtype=np.float)

# Question 4
w_mh = mh.log_metropolis_hastings(t_train, X_train[:, 4], mu_prior, sigma_square, N_samples)

plt.figure()
plt.plot(w_mh[0,:], w_mh[1,:],'o')
plt.title('Samples from log MH algorithm, N_s = ' + str(N_samples))
plt.xlabel('w0')
plt.ylabel('w1')
plt.savefig('Q4_weight_samples.png')

# One way to find an appropriate burn-in time and sample skip for de-correlation of samples
plt.figure()
plt.plot(np.arange(0, w_mh.shape[1]), w_mh[0, :], 'o')
plt.xlabel('#sample')
plt.ylabel('w0')
plt.figure()
plt.plot(np.arange(0, w_mh.shape[1]), w_mh[1, :], 'o')
plt.xlabel('#sample')
plt.ylabel('w1')

# Question 5
print("\n *** Selection 1 ***")
print("Making predictions from 5 column of X based on these parameters:")
print("mu_prior = " + str(mu_prior))
print("Prior sigma_square = " + str(sigma_square))
print("Number of samples = " + str(N_samples))

t_pred, t_pred_std = mh.prediction(X_test[:, 4], t_train, X_train[:, 4], mu_prior, sigma_square, N_samples)

fig, ax = plt.subplots()
ax.plot(X_test[:,4], t_test, 'bo', label='Test data')
ax.plot(X_test[:,4], t_pred, 'rx', label='Predictions')
#ax.errorbar(X_test[:,4], t_pred, t_pred_std, color='r', label='Predictions with errorbars')
plt.title('Test data and predictions, N_s = ' + str(N_samples))
plt.legend()
plt.xlabel('x')
plt.ylabel('t')
plt.savefig('Q5_model_1_predictions.png')

# TODO: Make the alternative scatter plot test t against predicted t
plt.figure()
plt.plot(t_test, t_pred, 'bo')
plt.xlabel('test t')
plt.ylabel('Predicted t')
plt.title('Alternative plot, Test data and predictions, N_s = ' + str(N_samples))
plt.savefig('Q5_alt_model_1_predictions.png')

print("RMSE = " + str(mh.RMSE(t_pred, t_test)))


# Selection 2 (features from t-2 and t-4)
mu_prior = np.array([1.0, 1.0, 1.0], dtype=np.float)

print("\n *** Selection 2 ***")
print("Making predictions from columns 3 and 4 of X based on these parameters:")
print("mu_prior = " + str(mu_prior))
print("Prior sigma_square = " + str(sigma_square))
print("Number of samples = " + str(N_samples))

t_pred, t_pred_std = mh.prediction(X_test[:, 2:4], t_train, X_train[:, 2:4], mu_prior, sigma_square, N_samples)

print("RMSE = " + str(mh.RMSE(t_pred, t_test)))


# Selection 3: Full model (features from t-1, t-2, t-4, t-8, and t-16)
mu_prior = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float)

print("\n *** Selection 3 ***")
print("Making predictions from all columns of X based on these parameters:")
print("mu_prior = " + str(mu_prior))
print("Prior sigma_square = " + str(sigma_square))
print("Number of samples = " + str(N_samples))

t_pred, t_pred_std = mh.prediction(X_test, t_train, X_train, mu_prior, sigma_square, N_samples)
print("RMSE = " + str(mh.RMSE(t_pred, t_test)))


# Show all figures
plt.show()
