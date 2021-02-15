# This is an implementation of the Metropolis-Hastings algorithm for sampling from
# a Poisson likelihood model of a linear regression model with a zero mean isotropic
# Gaussian prior

import numpy as np


def factorial_scalar(n):
    """This auxillary function allows for computing factorials of non-integers"""
    return np.prod(np.arange(1, np.math.floor(n)+1, dtype=np.uint64))

def factorial(n_vec):
    """This auxillary function allows for computing factorials of vectors non-integers"""
    f = np.vectorize(factorial_scalar)
    return f(n_vec)

def log_factorial_scalar(n):
    """This auxillary function allows for computing the logarithm of factorials of non-integers
       using Stirling's approximation."""
    n_floor = np.math.floor(n)
    if n_floor == 0:
        return 0.0
    else:
        return n_floor * np.log(n_floor) - n_floor  + 0.5 * np.log(2 * np.pi * n_floor)

def log_factorial(n_vec):
    """This auxillary function allows for computing logarithm of factorials of vectors of non-integers"""
    f = np.vectorize(log_factorial_scalar)
    return f(n_vec)


def linear_f(X, w):
    """Linear regression model with parameters w.
       We assume that X has shape (N,D) with N data points and D dimensions each
       and w has (D+1,) dimensions."""
    return np.dot(np.c_[np.ones((X.shape[0],1)), X], w)


def sample_prior(mu_prior, sigma_square, rng):
    """Returns a sample from the prior"""
    cov = sigma_square * np.eye(mu_prior.shape[0])
    return rng.multivariate_normal(mu_prior, cov, 1).reshape((mu_prior.shape[0],))

def prior(w, mu_prior, sigma_square = 1.0):
    """Zero mean isotropic Gaussian prior of the random parameter vector w with variance sigma_square.
       We assume that w has shape (D+1,)."""
    dim = w.shape[0]
    return np.exp(-((w-mu_prior)**2).sum() / (2*sigma_square)) / (np.sqrt(2 * np.pi * sigma_square)**dim)

def log_prior(w, mu_prior, sigma_square = 1.0):
    """Zero mean isotropic log-Gaussian prior of the random parameter vector w with variance sigma_square.
       We assume that w has shape (D+1,)."""
    dim = w.shape[0]
    return (-((w-mu_prior)**2).sum() / (2*sigma_square)) - np.log(np.sqrt(2 * np.pi * sigma_square)**dim)


def likelihood(w, t, X):
    """Poisson likelihood.
       We assume that X has shape (N,D) with N data points and D dimensions each,
       t has shape (N,), and w has (D+1,) dimensions.. """
    f = linear_f(X, w)
    #res = np.exp(np.log(f)*t) * np.exp(-f) / factorial(t)
    #return np.prod(res)
    res = np.sum(-f + np.log(f) * t - log_factorial(t))
    return np.exp(res)


def log_likelihood(w, t, X):
    """Poisson log-likelihood.
       We assume that X has shape (N,D) with N data points and D dimensions each,
       t has shape (N,), and w has (D+1,) dimensions.. """
    f = linear_f(X, w)
    return np.sum(-f + np.log(f) * t - log_factorial(t))


def posterior_unorm(w, t, X, mu_prior, sigma_square):
    return likelihood(w, t, X) * prior(w, mu_prior, sigma_square)

def log_posterior_unorm(w, t, X, mu_prior, sigma_square):
    return log_likelihood(w, t, X) + log_prior(w, mu_prior, sigma_square)


def acceptance_ratio(w_new, w_old, t, X, mu_prior, sigma_square):
    return posterior_unorm(w_new, t, X, mu_prior, sigma_square) / posterior_unorm(w_old, t, X, mu_prior, sigma_square)

def log_acceptance_ratio(w_new, w_old, t, X, mu_prior, sigma_square):
    return log_posterior_unorm(w_new, t, X, mu_prior, sigma_square) - log_posterior_unorm(w_old, t, X, mu_prior, sigma_square)


def generate_proposal(w_old, rng):
    """Samples a proposal w from the proposal distribution based on old value w_old.
       The proposal distribution is a Gaussian distribution centered in w_old."""
    sigma_square = 1.0 #0.5
    cov = sigma_square * np.eye(w_old.shape[0])
    return rng.multivariate_normal(w_old, cov, 1).reshape((w_old.shape[0],))



def log_metropolis_hastings(t, X, mu_prior, sigma_square, N_samples):
    """Metropolis-Hastings algorithm applied to log-acceptance ratio.
       Returns an array of (D+1) x N_samples samples."""

    # Allocate memory for the output array
    samples = np.zeros((mu_prior.shape[0], N_samples))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initial sample drawn from prior
    is_valid = False
    while not is_valid:
        w_current = sample_prior(mu_prior, sigma_square, rng)
        f_prior_sample = linear_f(X, w_current)
        if (np.all(f_prior_sample > 0)):
            is_valid = True
        else:
            print("Rejecting prior sample, try a new ...")


    for s in range(0, N_samples):

        # Generate a valid sample from the proposal distribution
        is_valid = False
        while not is_valid:
            w_proposal = generate_proposal(w_current, rng)
            # Not super-efficient - since a good proposal will lead to the evaluation of f on the
            # training set twice.

            f_proposal_sample = linear_f(X, w_proposal)
            if (np.all(f_proposal_sample > 0)):
                is_valid = True
            #else:
            #    print("Rejecting proposal sample (violating constraint on f), try a new ...")

        log_r = log_acceptance_ratio(w_proposal, w_current, t, X, mu_prior, sigma_square)
        if log_r >= 0.0:
            samples[:, s] = w_proposal
            w_current = w_proposal
        else:
            u = rng.uniform(0.0, 1.0, 1)
            if (np.log(u) <= log_r):
                samples[:, s] = w_proposal
                w_current = w_proposal
            else:
                samples[:, s] = w_current

    return samples


def prediction(Xnew, t, X, mu_prior, sigma_square, N_samples):
    """Make a prediction using MH samples from the posterior to estimate the expectation in the predictive
       distribution.
       Returns a predicted tnew as the average of the model f(x,w_s) evaluated for all samples of w_s."""

    # Skip samples in order to decorrelate the samples producing independent samples - find skip step
    burnin = 160 # Samples
    skipsteps = 10 # skip samples to get decorrelated samples
    w_mh_tmp = log_metropolis_hastings(t, X, mu_prior, sigma_square, (skipsteps * N_samples+burnin))
    w_mh = w_mh_tmp[:, burnin::skipsteps]

    tvec = linear_f(Xnew, w_mh)
    return np.mean(tvec, axis=1), np.std(tvec, axis=1)


def RMSE(t_pred, t_true):
    """Returns the RMSE"""
    return np.sqrt(np.mean((t_true - t_pred)**2))

# Test code
if (__name__=='__main__'):

    import matplotlib.pyplot as plt

    # Experimenting

    rng = np.random.default_rng()


    mu_prior = np.array([2.0, 0.5], dtype=np.float)
    sigma_square = 1.0

    w_true = np.random.rand(2) # uniform values between 0 and 1
    #w = np.zeros(6)
    #w[0] = 10.0


    # Generate some training data

    N_samples = 100

    X = rng.uniform(0.0, 160.0, (N_samples, 1))
    f = linear_f(X, w_true)
    pois_vec = np.vectorize(rng.poisson)
    t = pois_vec(f, 1)

    # Generate some test data
    N_test = 10
    X_test = rng.uniform(0.0, 160.0, (N_test, 1))
    f_test = linear_f(X_test, w_true)
    t_test = pois_vec(f_test, 1)

    plt.figure()
    plt.hist(t)
    plt.xlabel('t values')
    plt.ylabel('hist(t)')

    plt.figure()
    plt.plot(X, t, 'o')
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('t')

    plt.figure()
    plt.plot(X, f, 'o')
    plt.title('Linear data')
    plt.xlabel('x')
    plt.ylabel('f')

    X_train = X
    t_train = t

    print("True w = " + str(w_true))

    # sample another w
    #w_sample = generate_proposal(mu_prior, rng)
    w_sample = sample_prior(mu_prior, sigma_square, rng)
    print("Sample w = " + str(w_sample))

    print("Prior = " + str(prior(w_sample, mu_prior, sigma_square)))
    print("likelihood = " + str(likelihood(w_sample, t_train, X_train)))
    print("Unormalized posterior = " + str(posterior_unorm(w_sample, t_train, X_train, mu_prior, sigma_square)))
    print("Acceptance ratio = " + str(acceptance_ratio(w_sample, w_true, t_train, X_train, mu_prior, sigma_square)))

    print("Log Prior = " + str(log_prior(w_sample, mu_prior, sigma_square)))
    print("Log likelihood = " + str(log_likelihood(w_sample, t_train, X_train)))
    print("Log Unormalized posterior = " + str(log_posterior_unorm(w_sample, t_train, X_train, mu_prior, sigma_square)))
    print("Log Acceptance ratio = " + str(log_acceptance_ratio(w_sample, w_true, t_train, X_train, mu_prior, sigma_square)))


    w_mh = log_metropolis_hastings(t_train, X_train, mu_prior, sigma_square, 1000)

    plt.figure()
    plt.plot(w_mh[0,:], w_mh[1,:],'o')
    plt.title('Samples from log MH algorithm')
    plt.xlabel('w0')
    plt.ylabel('w1')


    plt.figure()
    plt.plot(np.arange(0, w_mh.shape[1]), w_mh[0, :], 'o')
    plt.xlabel('#sample')
    plt.ylabel('w0')
    plt.figure()
    plt.plot(np.arange(0, w_mh.shape[1]), w_mh[1, :], 'o')
    plt.xlabel('#sample')
    plt.ylabel('w1')


    # Make predictions based on the model
    t_pred, t_pred_std = prediction(X_test, t_train, X_train, mu_prior, sigma_square, 1000)

    fig, ax = plt.subplots()
    ax.plot(X_test, t_test, 'bo', label='Test data')
    ax.plot(X_test, t_pred, 'rx', label='Predictions')
    ax.errorbar(X_test, t_pred, t_pred_std, color='r', label='Predictions with errorbars')
    plt.title('Test data and predictions')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('t')

    print("RMSE = " + str(RMSE(t_pred, t_test)))


    plt.show()