import numpy as np

def var_g(ga,gb, ga_mean = None, gb_mean = None):
    
    if isinstance(ga,(np.ndarray)) == False:
        ga = np.array(ga)
    if isinstance(gb,(np.ndarray)) == False:
        gb = np.array(gb)
    
    if ga_mean == None:
        ga_mean = np.mean(ga)

    if gb_mean == None:
        gb_mean = np.mean(gb)

    if ga.shape != gb.shape:
        raise Exception("The length of G(A)_i is not equal to the length of G(B)_i!")
    
    n = ga.shape[0]
    var_g = 1/(2*n)*np.sum((ga - ga_mean)**2 + (gb - gb_mean)**2)

    return var_g


def sj_times_var_g(ga,gb,gj):
    
    if isinstance(ga,(np.ndarray)) == False:
        ga = np.array(ga)
    if isinstance(gb,(np.ndarray)) == False:
        gb = np.array(gb)
    if isinstance(gj,(np.ndarray)) == False:
        gj = np.array(gj)
    
    if (ga.shape != gb.shape) or (ga.shape != gj.shape):
        raise Exception("The length of G(A)_i is not equal to the length of G(B)_i!")
    
    n = ga.shape[0]
    gj_ga_factor = gj-ga
    sj_times_var_g = 1/(n)*np.sum(np.multiply(gb,gj_ga_factor))

    return sj_times_var_g

def sj(ga,gb,gj,ga_mean = None, gb_mean = None):
    if ga_mean == None:
        ga_mean = np.mean(ga)
    if gb_mean == None:
        gb_mean = np.mean(gb)
    
    var_g_val = var_g(ga,gb, ga_mean, gb_mean)
    sj_times_var_g_val = sj_times_var_g(ga,gb,gj)
    if abs(var_g_val) < 0.000001:
        return 0
    
    sj_val = sj_times_var_g_val/var_g_val
    
    return sj_val

def sj_tot_times_var_g(ga,gj):
    if isinstance(ga,(np.ndarray)) == False:
        ga = np.array(ga)
    if isinstance(gj,(np.ndarray)) == False:
        gj = np.array(gj)

    if ga.shape != gj.shape:
        raise Exception("The length of G(A)_i is not equal to the length of G(B)_i!")
    
    n = ga.shape[0]

    sj_tot_times_var_g = 1/(2*n)*np.sum((ga-gj)**2)

    return sj_tot_times_var_g

def sj_tot(ga,gb,gj, ga_mean = None, gb_mean = None):
    if ga_mean == None:
        ga_mean = np.mean(ga)
    if gb_mean == None:
        gb_mean = np.mean(gb)
    
    var_g_val = var_g(ga,gb, ga_mean, gb_mean)
    sj_tot_times_var_g_val = sj_tot_times_var_g(ga,gj)
    if abs(var_g_val) < 0.000001:
        return 0

    sj_tot_val = sj_tot_times_var_g_val/var_g_val

    return sj_tot_val
