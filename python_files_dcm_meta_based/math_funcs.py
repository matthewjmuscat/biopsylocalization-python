import numpy as np
import math 
from statsmodels.nonparametric import kernel_regression
import statsmodels.api as sm

def binomial_likelihood(p, num_trials, num_successes):
    # note that we omit the factor out front as it does not depend on the probability estimator p
    n = num_trials
    y = num_successes
    likelihood = (p**y)*((1-p)**(n-y))
    return likelihood

def binomial_log_likelihood(p, num_trials, num_successes):
    # note that we omit the factor out front as it does not depend on the probability estimator p
    n = num_trials
    y = num_successes
    log_likelihood = y*math.log(p)+(n-y)*math.log(1-p)
    return log_likelihood

def binomial_log_likelihood_2nd_der(p, num_trials, num_successes):
    # note that we omit the factor out front as it does not depend on the probability estimator p
    # evaluating this at the probability estimator leads to finding the variance of the probability estimator when taking the negative inverse
    n = num_trials
    y = num_successes
    if p == 0. or p == 1.:
        log_likelihood_2nd_der = 0.
    else:
        log_likelihood_2nd_der = (-y/(p**2)) - ((n-y)/(1-p)**2)
    return log_likelihood_2nd_der

def binomial_variance_estimator(probability_estimator, num_trials, num_successes):
    p_hat = probability_estimator
    n = num_trials
    y = num_successes
    log_likelihood_2nd_der = binomial_log_likelihood_2nd_der(p_hat, n, y)
    if log_likelihood_2nd_der == 0.:
        p_hat_variance = 0.
    else:
        p_hat_variance = -1/log_likelihood_2nd_der
    return p_hat_variance

def binomial_se_estimator(probability_estimator, num_trials, num_successes):
    p_hat = probability_estimator
    n = num_trials
    y = num_successes
    p_hat_variance = binomial_variance_estimator(p_hat, n, y)
    p_hat_se = math.sqrt(p_hat_variance)
    return p_hat_se

def binomial_CI_estimator(probability_estimator, num_trials, num_successes):
    p_hat = probability_estimator
    n = num_trials
    y = num_successes
    z0975=1.96
    p_hat_se = binomial_se_estimator(p_hat, n, y)
    p_hat_CI_lower = p_hat - z0975*p_hat_se
    p_hat_CI_upper = p_hat + z0975*p_hat_se
    p_hat_CI_tuple = (p_hat_CI_lower,p_hat_CI_upper)
    return p_hat_CI_tuple


def normal_mean_se_var_estimatation(data_1d_arr):
    mean_estimator = np.mean(data_1d_arr)
    mu = mean_estimator
    num_trials = data_1d_arr.shape[0]
    n = num_trials

    sum_terms_arr = (data_1d_arr - mean_estimator)**2
    sum_terms_arr_summed = np.sum(sum_terms_arr)
    mle_se = math.sqrt((1/n)*sum_terms_arr_summed)
    mle_var = mle_se**2
    mu_se_var_tuple = (mu,mle_se,mle_var)

    return mu_se_var_tuple


def normal_CI_estimator(mu_estimator, se_estimator):
    mu_hat = mu_estimator
    se_hat = se_estimator
    z0975=1.96
    mu_hat_CI_lower = mu_hat - z0975*se_hat
    mu_hat_CI_upper = mu_hat + z0975*se_hat
    mu_hat_CI_tuple = (mu_hat_CI_lower,mu_hat_CI_upper)
    return mu_hat_CI_tuple




def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def lowess_with_confidence_bounds(
    x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top



def non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
    parallel_pool,
    x, y, eval_x, 
    N=200, conf_interval=0.95, bandwidth = 0.5
):
    x = np.asarray(x)
    y = np.asarray(y)
    args_list = [float('Nan')]*(N+1)
    args_list[0] = (x, y, eval_x)
    for i in range(1,N+1):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]
        args_list[i] = (sampled_x, sampled_y, eval_x, bandwidth)
    
    all_NPKR_fit_vals_list = parallel_pool.starmap(non_param_kernel_regression, args_list)
    NPKR_fit_vals = all_NPKR_fit_vals_list[0]
    bootstrapped_NPKR_fit_vals_list = all_NPKR_fit_vals_list[1:]
    bootstrapped_NPKR_fit_vals_arr = np.asarray(bootstrapped_NPKR_fit_vals_list)

    # Get the confidence interval
    sorted_values = np.sort(bootstrapped_NPKR_fit_vals_arr, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    if bound == 0:
        bottom = sorted_values[0]
        top = sorted_values[-1]
    else:
        bottom = sorted_values[bound - 1]
        top = sorted_values[-bound]

    return NPKR_fit_vals, bottom, top
    

def non_param_kernel_regression(x, y, eval_x, bandwidth = 0.5, NPKR_type='ll'):
    NPKR_class_obj = kernel_regression.KernelReg(exog=x, endog=y, var_type='c', reg_type = NPKR_type, bw = [bandwidth])
    NPKR_fit_vals, NPKR_partial_derivatives_vals = NPKR_class_obj.fit(eval_x)
    # Note that I do not return the partial derivatives as they do not seem important here
    return NPKR_fit_vals


def non_param_kernel_regression_with_confidence_bounds_bootstrap(
    x, y, eval_x, N=200, conf_interval=0.95, NPKR_type='ll'
):
    """
    Perform NPKR (LL) regression and determine a confidence interval by bootstrap resampling
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # NPKR smoothing
    NPKR_class_obj = kernel_regression.KernelReg(exog=x, endog=y, var_type='c', reg_type = NPKR_type)
    NPKR_fit_vals = NPKR_class_obj.fit(eval_x)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    NPKR_bootstrapped_fits = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        NPKR_bootstrapped_class_obj = kernel_regression.KernelReg(exog=sampled_x, endog=sampled_y, var_type='c', reg_type = NPKR_type)
        NPKR_bootstrapped_fits[i] = NPKR_bootstrapped_class_obj.fit(eval_x)

    # Get the confidence interval
    sorted_values = np.sort(NPKR_bootstrapped_fits, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return NPKR_fit_vals, bottom, top



def non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
    parallel_pool,
    x, y, eval_x, 
    N=200, conf_interval=0.95
):
    x = np.asarray(x)
    y = np.asarray(y)
    args_list = [float('Nan')]*(N+1)
    args_list[0] = (x, y, eval_x)
    for i in range(1,N+1):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]
        args_list[i] = (sampled_x, sampled_y, eval_x)
    
    all_NPLR_fit_vals_list = parallel_pool.starmap(non_param_LOWESS_regression, args_list)
    NPKR_fit_vals = all_NPLR_fit_vals_list[0]
    bootstrapped_NPKR_fit_vals_list = all_NPLR_fit_vals_list[1:]
    bootstrapped_NPKR_fit_vals_arr = np.asarray(bootstrapped_NPKR_fit_vals_list)

    # Get the confidence interval
    sorted_values = np.sort(bootstrapped_NPKR_fit_vals_arr, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    if bound == 0:
        bottom = sorted_values[0]
        top = sorted_values[-1]
    else:
        bottom = sorted_values[bound - 1]
        top = sorted_values[-bound]

    return NPKR_fit_vals, bottom, top
    

def non_param_LOWESS_regression(x, y, eval_x):
    NPLR_fit_vals = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x)
    return NPLR_fit_vals