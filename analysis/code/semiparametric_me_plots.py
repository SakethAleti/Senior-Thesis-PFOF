import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from openpyxl import Workbook, load_workbook, utils
from openpyxl.styles import Alignment, Font
import string
from matplotlib2tikz import save as tikz_save
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#### Functions

def trim(X, percent):
    ''' Trims X with the percent input by finding the top and bottom
    (percent/2) observations and then returns a matrix of binary variables
    indicating rows of X that would not be trimmed
    '''

    alpha = (1 - percent)/2
    n, k = np.shape(X)
    t_ind = np.zeros((n, k))

    for i in range(0, k):
        upper_bd = np.percentile(X[:,i], (1 - alpha)*100)
        lower_bd = np.percentile(X[:,i], (alpha)*100)
        t_ind[:, i] = [int(lower_bd < x < upper_bd) for x in X[:,i]]

    return np.prod(t_ind, axis = 1)


def CE_1(Y, X, arg, r):
    ''' Conditional Expectation of Y given X at arg (matrix) with bandwith r
    using a gaussian kernel
    '''

    n_arg = np.shape(arg)[0]
    n = np.shape(X)[0]
    h = (n**(-r)) * np.std(X, axis = 0, ddof = 1)
    e = np.zeros((n_arg, 1))

    for j in range(0, n_arg):
        k = np.divide(norm.pdf(np.divide((arg[j] - X), h)), h)
        k = np.prod(k, axis = 1)
        e[j] = (Y.T*k/n)/np.mean(k)

    return e


def SLS_1(b, Y, X, X_ind, h = 1/5):
    ''' Semiparametric least-squares using CE_1 and bandwith = 1/5
    '''

    v = X * np.matrix(b).T
    EY = CE_1(Y, v, v, h)
    residual = np.power((Y - EY), 2)

    return (-0.5 * np.matrix(X_ind)*residual)


# Regress Y on X with SLS
def run_semiparametric_regression(Y, X, guess, trim_percent = 0.98,
    xtol = 0.001, maxiter = 1):
    ''' Runs SLS with some default parameters, approximates an initial guess
    using BFGS
    '''

    obj_f = lambda x_0: -1e6*SLS_1(np.append(np.array([1]), x_0), Y, X,
        trim(X, 0.98))[0,0]

    print('    Running LS...')
    result = least_squares(obj_f, list(np.array(guess).flatten()), xtol = xtol)

    print('    BFGS...')
    result = minimize(obj_f, list(np.array(guess).flatten()), method='BFGS',
        options = {'maxiter': maxiter})

    return result


def convert_hessian_to_cov(Y, X, results):
    ''' Converts the result output from scipy.optimize into a covariance matrix
    by taking the inverse of the hessian and multiplying by an estimate of the
    variance of the residuals
    '''

    sigma_2_hat = np.mean(np.power(Y - X*np.matrix(results.x).T, 2))
    return results.hess_inv * sigma_2_hat


def compute_marginal_effect(Y, X, ind, point, beta, delta = 0.01, h = 1/5):
    ''' Finds the marginal effects at a given point using CE_1, delta
    represents the amount to nudge the point by when calculating the marginal
    effects, ind refers to the index of the variable being nudged
    '''

    point_nudge = np.copy(point)
    point_nudge[0, ind] = point_nudge[0, ind] + delta
    point_nudge = np.matrix(point_nudge)

    v_hat = X*beta
    v_hat_avg = point*beta
    v_hat_avg_nudge = point_nudge*beta

    return np.asscalar(CE_1(Y, v_hat, v_hat_avg_nudge, h) -
        CE_1(Y, v_hat, v_hat_avg, h))/delta


def find_tstats(Y, X, results):
    ''' Computes t_stats using an input of variables and the results
    of a scipy.minimize routine (must output a hessian)
    '''

    V = convert_hessian_to_cov(Y, X, results)

    n = np.shape(results.x)[0]
    theta = results.x/results.x[0]
    t_stats = np.zeros(shape = (n))
    t_stats[0] = np.nan # first t-stat is unknown

    for i in range(1, n):
        t_stats[i] = theta[i] / np.sqrt(V[i,i])

    return t_stats

def get_sig_stars(coeff, stderr, p_value_labels):
    ''' Outputs significance stars after calculating the t-stat
    '''
    t_stat = coeff/stderr
    p_val  = 2*(1 - norm.cdf(np.abs(t_stat)))
    below_ind = np.where([p_val < x for x in p_value_labels.keys()])[0]
    below_vals = [list(p_value_labels.keys())[i] for i in below_ind]
    if not below_vals:
        return ''
    else:
        min_p_val = np.min(below_vals)
        return p_value_labels[min_p_val]

def CE_m(X, arg, beta, r):
    ''' Conditional Expectation of g_m(v_bar) with bandwith r
    using a gaussian kernel: \sum K((v_bar - v_i) / h) (1/Nh)
    '''

    n_arg = np.shape(arg)[0]
    n = np.shape(X)[0]
    v = X*beta
    v_bar = arg*beta
    h = (n**(-r)) * np.std(v, axis = 0, ddof = 1)

    k = np.divide(norm.pdf(np.divide((v_bar - v), h)), h)
    return np.sum(k, axis = 0)/n

def ME_var(Y, X, arg, beta, r):
    ''' Compute variance of a marginal effect'''

    # marginal density of v_bar
    g_m = CE_m(X, arg, beta, r)

    # integral y^2 g*(y | v_bar) dy
    temp = CE_1(np.power(Y,2), X*beta, arg*beta, r)

    # var of me
    n = np.shape(X)[0]
    return g_m*temp*(1/np.sqrt(2))/(n*r)

def plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend,
    plot_figsize = (7.5,5), legend_loc = 'best', ylabel = 'Effect on Market Share'):
    ''' Plotting function for marginal effects with pre-defined parameters to fit LaTeX file.
    '''

    fig = plt.figure(figsize=plot_figsize);

    # first fit
    plt.plot(plot_data_1[:,0], plot_data_1[:,1], color = (0, 0, 0.7))
    plt.fill_between(plot_data_1[:,0], plot_data_1[:,1] + 2*plot_data_1[:,2], plot_data_1[:,1] - 2*plot_data_1[:,2], color = (0.1,0.1,0.1), alpha = 0.05)

    # second fit
    plt.plot(plot_data_2[:,0], plot_data_2[:,1], color = (0.6, 0, 0))
    plt.fill_between(plot_data_2[:,0], plot_data_2[:,1] + 2*plot_data_2[:,2], plot_data_2[:,1] - 2*plot_data_2[:,2], color = (0.1,0.1,0.1), alpha = 0.05)

    # average me's
    plt.plot([np.min(plot_data_1[:,0]), np.max(plot_data_1[:,0])], [plot_data_1_avg, plot_data_1_avg], linestyle = '--', color = (0, 0, 0.7))
    plt.plot([np.min(plot_data_2[:,0]), np.max(plot_data_2[:,0])], [plot_data_2_avg, plot_data_2_avg], linestyle = '--', color = (0.6, 0, 0))

    plt.title(plot_title)
    plt.ylabel(ylabel)
    plt.xlabel('Execution Quality Percentile')
    plt.legend(plot_legend, loc = legend_loc)
    plt.grid('on', color = (0.8, 0.8, 0.8))
    plt.tight_layout()

    return fig




#### Main



### Prelims


## Formulae

# Fits
fit1_formula = 'MktShare ~ PrImp_Pct + PrImp_AvgAmt + PrImp_AvgT'
fit2_formula = 'MktShare ~ PrImp_ExpAmt + PrImp_AvgT'
fit3_formula = 'MktShare ~ PrImp_Pct + PrImp_AvgAmt + All_AvgT'
fit4_formula = 'MktShare ~ PrImp_ExpAmt + All_AvgT'

formulaCols = lambda x: x.replace(' ', '').replace('~', '+').split('+')
fit_formulae = [fit1_formula, fit2_formula, fit3_formula, fit4_formula]
fit_formulae = [formulaCols(x) for x in fit_formulae]


## Initialize empty fit objects
fit_results_paid = [None]*4
fit_results_unpaid  = [None]*4

for i in range(0,4):
    fit_results_paid[i]   = sp.optimize.optimize.OptimizeResult()
    fit_results_paid[i].x = []
    fit_results_unpaid[i]   = sp.optimize.optimize.OptimizeResult()
    fit_results_unpaid[i].x = []


## Enter coefficients into fit objects

# Values are from STATA results from semiparametric_model.do
# Using STATA regression results is faster than rerunning SLS in python
fit_results_paid[0].x = np.array([1, 1.02, 0.000771])
fit_results_paid[1].x = np.array([1, -0.00995])
fit_results_paid[2].x = np.array([1, 1.061, 0.000962])
fit_results_paid[3].x = np.array([1, -0.00114])

fit_results_unpaid[0].x = np.array([1, -0.249, -0.143])
fit_results_unpaid[1].x = np.array([1, -0.0462])
fit_results_unpaid[2].x = np.array([1, 0.711, -0.00674])
fit_results_unpaid[3].x = np.array([1, -0.0307])

# used for the delta method when calculating marginal effects
delta = 10**-7
# bandwidth size
h = 0.002


## Load data
fit_marginal_effects = [None]*len(fit_formulae)
data_df = pd.read_csv('../../data/processed/regression_data_levels.csv').query('OrderType == "Market" & MktShare != 0')



### Marginal Effects Calculation


## Calculate difference in marginal effects between Non-POF and POF Brokers
for i in range(0, len(fit_formulae)):

    data = data_df[fit_formulae[i]]
    data_pof  = data_df.query('Rebate_Dummy == 1')[fit_formulae[i]]
    data_npof = data_df.query('Rebate_Dummy == 0')[fit_formulae[i]]

    X = np.matrix(data)[:, 1:]
    X_pof = np.matrix(data_pof)[:, 1:]
    X_npof = np.matrix(data_npof)[:, 1:]

    Y = np.matrix(data)[:, 0]
    Y_pof = np.matrix(data_pof)[:, 0]
    Y_npof = np.matrix(data_npof)[:, 0]

    fit_marginal_effects[i] = {}

    for j in range(len(fit_results_paid[i].x)):

        temp_results = np.empty((0,3))

        for percentile in np.linspace(10, 91, 60):

            X_percentile = np.percentile(X, percentile, axis = 0)
            # Last coefficient is for average time which is a negative measure
            X_percentile[-1] = np.percentile(X[:,-1], 100 - percentile, axis = 0)

            paid_me   = compute_marginal_effect(Y_pof, X_pof, j, np.matrix(X_percentile), np.matrix(fit_results_paid[i].x).T,   delta = delta, h = h)
            unpaid_me = compute_marginal_effect(Y_npof, X_npof, j, np.matrix(X_percentile), np.matrix(fit_results_unpaid[i].x).T, delta = delta, h = h)
            me_var_total = ME_var(Y_pof, X_pof, X_percentile, np.matrix(fit_results_paid[i].x).T, 1/5) + \
                            ME_var(Y_npof, X_npof, X_percentile, np.matrix(fit_results_unpaid[i].x).T, 1/5)


            temp_results = np.append(temp_results,
                                     [[percentile, unpaid_me-paid_me, me_var_total]],
                                     axis = 0)

        fit_marginal_effects[i][fit_formulae[i][j+1]] = temp_results


## Calculate Average Difference in Marginal Effects between Non-POF and POF

fit_marginal_effects_avg = [None]*len(fit_formulae)

for i in range(0, len(fit_formulae)):

    print(i)

    data = data_df[fit_formulae[i]]
    data_pof  = data_df.query('Rebate_Dummy == 1')[fit_formulae[i]]
    data_npof = data_df.query('Rebate_Dummy == 0')[fit_formulae[i]]

    X = np.matrix(data)[:, 1:]
    X_pof = np.matrix(data_pof)[:, 1:]
    X_npof = np.matrix(data_npof)[:, 1:]

    Y = np.matrix(data)[:, 0]
    Y_pof = np.matrix(data_pof)[:, 0]
    Y_npof = np.matrix(data_npof)[:, 0]

    fit_marginal_effects_avg[i] = {}

    for j in range(len(fit_results_paid[i].x)):

        temp_results = np.empty((0,2))
        temp_results_paid = []
        temp_results_unpaid = []

        for k in range(0, int(X_pof.shape[0]/10)):

            paid_me   = compute_marginal_effect(Y_pof, X_pof, j, X_pof[k,:], np.matrix(fit_results_paid[i].x).T,   delta = delta, h = h)
            temp_results_paid = temp_results_paid + [paid_me]

        for k in range(0, int(X_npof.shape[0]/10)):

            unpaid_me = compute_marginal_effect(Y_npof, X_npof, j, X_npof[k,:], np.matrix(fit_results_unpaid[i].x).T, delta = delta, h = h)
            temp_results_unpaid = temp_results_unpaid  + [unpaid_me]

        fit_marginal_effects_avg[i][fit_formulae[i][j+1]] = np.mean(temp_results_unpaid) - np.mean(temp_results_paid)



### Marginal Effect Plots


## Set up plot params
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.variant"] = "small-caps"
plt.rcParams["font.size"] = 14

# Empty dict of marginal effect figure objects
me_figs = {}


## Plots

# Average Price Improvement
plot_var = 'PrImp_AvgAmt'
plot_fits = [0, 2]
plot_title  = 'Average Price Improvement'
plot_legend = ['Fit ' + str(plot_fits[0] + 1), 'Fit ' + str(plot_fits[1] + 1)]
plot_data_1 = fit_marginal_effects[plot_fits[0]][plot_var]
plot_data_2 = fit_marginal_effects[plot_fits[1]][plot_var]
plot_data_1_avg = fit_marginal_effects_avg[plot_fits[0]][plot_var]
plot_data_2_avg = fit_marginal_effects_avg[plot_fits[1]][plot_var]

fig = plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend)
me_figs[plot_var] = fig

# Percent Price Improvement
plot_var = 'PrImp_Pct'
plot_fits = [0, 2]
plot_title  = 'Percent Price-Improved'
plot_legend = ['Fit ' + str(plot_fits[0] + 1), 'Fit ' + str(plot_fits[1] + 1)]
plot_data_1 = fit_marginal_effects[plot_fits[0]][plot_var]
plot_data_2 = fit_marginal_effects[plot_fits[1]][plot_var]
plot_data_1_avg = fit_marginal_effects_avg[plot_fits[0]][plot_var]
plot_data_2_avg = fit_marginal_effects_avg[plot_fits[1]][plot_var]

fig = plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend)
me_figs[plot_var] = fig

# Expected Price Improvement
plot_var = 'PrImp_ExpAmt'
plot_fits = [1, 3]
plot_title  = 'Expected Price Improvement'
plot_legend = ['Fit ' + str(plot_fits[0] + 1), 'Fit ' + str(plot_fits[1] + 1)]
plot_data_1 = fit_marginal_effects[plot_fits[0]][plot_var]
plot_data_2 = fit_marginal_effects[plot_fits[1]][plot_var]
plot_data_1_avg = fit_marginal_effects_avg[plot_fits[0]][plot_var]
plot_data_2_avg = fit_marginal_effects_avg[plot_fits[1]][plot_var]

fig = plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend)
me_figs[plot_var] = fig

# Average Exec Time for Price-Improved
plot_var = 'PrImp_AvgT'
plot_fits = [0, 1]
plot_title  = 'Execution Speed (Price Imp. Shares)'
plot_legend = ['Fit ' + str(plot_fits[0] + 1), 'Fit ' + str(plot_fits[1] + 1)]
plot_data_1 = fit_marginal_effects[plot_fits[0]][plot_var]
plot_data_2 = fit_marginal_effects[plot_fits[1]][plot_var]
plot_data_1_avg = fit_marginal_effects_avg[plot_fits[0]][plot_var]
plot_data_2_avg = fit_marginal_effects_avg[plot_fits[1]][plot_var]

fig = plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend)
me_figs[plot_var] = fig

# Average Exec Time for All
plot_var = 'All_AvgT'
plot_fits = [2, 3]
plot_title  = 'Execution Speed (All Shares)'
plot_legend = ['Fit ' + str(plot_fits[0] + 1), 'Fit ' + str(plot_fits[1] + 1)]
plot_data_1 = fit_marginal_effects[plot_fits[0]][plot_var]
plot_data_2 = fit_marginal_effects[plot_fits[1]][plot_var]
plot_data_1_avg = fit_marginal_effects_avg[plot_fits[0]][plot_var]
plot_data_2_avg = fit_marginal_effects_avg[plot_fits[1]][plot_var]

fig = plot_me(plot_data_1, plot_data_2, plot_data_1_avg, plot_data_2_avg, plot_title, plot_legend)
me_figs[plot_var] = fig


## Save plots as tex files

for fig_name, fig in me_figs.items():
    tikz_save('../../exhibits/marginaleffects/' + fig_name + '.tex', figure = fig,
              textsize=8.0, figurewidth= '3in', figureheight = '2in', show_info = False,
              extra_axis_parameters =  set(['align = center', 'scaled ticks = false']))
