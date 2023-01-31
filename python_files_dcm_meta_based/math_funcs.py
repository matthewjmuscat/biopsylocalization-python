import numpy as np
import math 

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