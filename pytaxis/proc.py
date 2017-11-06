#!======Open librairies================
import numpy as np
import pandas as pd
from sklearn import mixture
from hmmlearn import hmm
from pandas import DataFrame, Series
from scipy import stats, integrate
import itertools
import pylab
import matplotlib.pyplot as plt
#!==============Define functions================
#!=====================================================================================
def find_tumbles(traj,  model, params = ['vel_norm', 'acc_norm', 'acc_angle'], 
                    n_components = 3, threshold = 1,
                    covariance_type = 'diag',
                    n_iter = 1000,
                    model_type = 'HMM',
                    int_state = 1):
    i = 0
    run_state = 1
    plt.figure(figsize=(6,5))
    tol = 100
    traj = calc_stat(traj, ['vel'], [lambda x: np.nanpercentile(x, 95)], ['run_mean'])
    
    while tol > threshold:
        print 'iteration ', i
        i += 1
        traj = traj[traj.vel_run_mean != 0]
        traj.loc[:, 'vel_norm'] = traj.loc[:, 'vel']/traj.loc[:, 'vel_run_mean']
        traj.loc[:, 'acc_norm'] = traj.loc[:, 'acc']/traj.loc[:, 'vel_run_mean']
        if 'vel_run_mean_old' in traj.keys():
            traj.drop('vel_run_mean_old', inplace = True, axis = 1)
        traj.rename(columns = {'vel_run_mean': 'vel_run_mean_old'}, inplace=True)

        model = fit_model(traj, params, n_components, 
                          covariance_type,
                          n_iter, 
                          model_type,
                          model)
        _, _, _ = assign_tumbles(traj, 'vel', state_type = 'state_' + model_type, tbias_type = 'tbias_' + model_type, int_state = int_state)
        
        traj = calc_run_stat(traj, param = 'vel', state = 'tbias_' + model_type)
        
        traj_stats = traj[['particle', 
                            'vel_run_mean', 
                            'vel_run_mean_old']].groupby([u'particle'], 
                                                           as_index = False).agg(np.nanmean)

        tol = np.mean(abs(traj_stats.vel_run_mean - traj_stats.vel_run_mean_old)/traj_stats.vel_run_mean_old)*100
        plt.plot(i, tol, 'o')
    plt.xlabel('iteration')
    plt.ylabel('% change')
    plt.xlim(0, i + 1)
    return model, traj
#!=====================================================================================
def assign_tumbles(traj, vel_type, state_type, tbias_type, int_state):
    vel_mean = np.zeros(len(traj[state_type].unique()))
    for state in traj[state_type].unique():
        vel_mean[state] = traj.loc[traj[state_type] == state, vel_type].mean()
    tumble_state = np.where(vel_mean == min(vel_mean))[0][0]
    run_state = np.where(vel_mean == max(vel_mean))[0][0]
    traj.loc[:, tbias_type] = int_state
    traj.loc[traj[state_type] == tumble_state, tbias_type] = 0
    traj.loc[traj[state_type] == run_state, tbias_type] = 1
    return run_state, tumble_state, vel_mean
#!=====================================================================================
def fit_model(traj, params, n_components, 
              covariance_type,
              n_iter, 
              model_type,
              model):
    X = np.stack((traj[param] for param in params), -1)
    lengths = traj[['particle', 'frame']].groupby([u'particle'], as_index = False).count().frame.values
    
    if not model:
        print 'No model provided, constructing {} model based on the current dataset'.format(model_type)
        if model_type == 'HMM':
            model = hmm.GaussianHMM(n_components = n_components, covariance_type = covariance_type, n_iter = n_iter).fit(X, lengths)
        if model_type == 'GMM':
            model = mixture.GaussianMixture(n_components = n_components, covariance_type = covariance_type, max_iter = n_iter).fit(X)
        if model_type == 'GMM_Dirichlet':
            model = mixture.BayesianGaussianMixture(n_components = n_components, covariance_type = covariance_type, max_iter = n_iter, 
                                                    weight_concentration_prior_type='dirichlet_distribution').fit(X)
    state = model.predict(X)
    traj.loc[:, 'state_' + model_type] = state
    return model
#!=====================================================================================
def calc_stat(dataset, params, funcs, col_names):
    keys = [u'particle'] + params
    grouped = dataset[keys].groupby([u'particle'], as_index = False)
    dataset_stats = grouped.agg(funcs)
    
    new_names = list(itertools.chain(*[col_names for i in range(len(keys))]))
    dataset_stats.columns.set_levels(new_names, level=1, inplace=True)
    
    dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns.values]
    
    cols2drop = list(set(dataset_stats.columns) & set(dataset.keys()))
    dataset.drop(cols2drop, inplace = True, axis = 1)
    
    dataset_stats.reset_index(level = 0, inplace='True')
    dataset = pd.merge(dataset, dataset_stats, on = ['particle'])
    return dataset
#!=====================================================================================
def calc_run_stat(traj, param, state):
    traj.loc[:, param + '_run'] = traj.loc[:, param]
    traj.loc[traj[state] == 0, param + '_run'] = None
    if param + '_run_mean' in traj.keys():
        traj.drop(param + '_run_mean', inplace = True, axis = 1)
        
    traj = calc_stat(traj, [param + '_run'], [np.nanmean], ['mean'])
    return traj
#!=====================================================================================
def calc_params(traj, wind, fps, pix_size):
    left_edge, right_edge, idx_edge, _, _ = find_edges(traj, wind)
    
    vel_x = calc_diff(traj, 'x', wind, left_edge, right_edge, idx_edge)*fps*pix_size
    vel_y = calc_diff(traj, 'y', wind, left_edge, right_edge, idx_edge)*fps*pix_size
    traj.loc[:, 'vel'] = (vel_x**2 + vel_y**2)**0.5
    traj.loc[:, 'acc'] = calc_diff(traj, 'vel', wind, left_edge, right_edge, idx_edge)*fps
    traj.loc[:, 'angle'] = np.arctan2(vel_y, vel_x)
    
    angle_change = calc_diff(traj, 'angle', wind, left_edge, right_edge, idx_edge)
    
    traj.loc[:, 'vel_angle'] = np.abs(np.mod(angle_change - pylab.pi, pylab.pi*2) - pylab.pi)*fps
    traj.loc[:, 'acc_angle'] = calc_diff(traj, 'vel_angle', wind, left_edge, right_edge, idx_edge)*fps
    
#!=====================================================================================
def calc_diff(traj, param, wind, left_edge, right_edge, idx_edge, dividebywind = False):
    
    diff = (traj[param].values[wind:] - traj[param].values[:-wind])
    if dividebywind == True:
        diff = diff/wind
    else:
        diff = diff/(traj['frame'].values[wind:] - traj['frame'].values[:-wind])
    diff = np.concatenate((np.repeat(np.nan, left_edge), diff, np.repeat(np.nan, right_edge)))
    diff[idx_edge] = np.nan
    return diff
#!=====================================================================================
def find_edges(traj, wind):
    traj.reset_index(drop = True, inplace = True)
    traj.loc[:, 'index'] = traj.index
    idx_start = traj.groupby('particle').agg('first')['index'].values
    idx_end = traj.groupby('particle').agg('last')['index'].values

    left_edge = wind/2
    right_edge = wind - left_edge
    idx_edge = [np.concatenate((range(idx_start[i], idx_start[i] + left_edge), range(idx_end[i], idx_end[i] - right_edge, -1)))\
               for i in range(len(idx_end))]
    idx_edge = list(itertools.chain(*idx_edge))
    return left_edge, right_edge, idx_edge, idx_start, idx_end

#!=====================================================================================
def find_MADs_KDE(x, y):
    
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel.evaluate(positions).T, X.shape)
    extent = [xmin, xmax, ymin, ymax]
    
    x0 = positions[0][np.argmax(Z)]
    y0 = positions[1][np.argmax(Z)]

    MADx = np.median(abs(x - x0))
    MADy = np.median(abs(y - y0))

    return (x0, y0), (MADx, MADy), (Z, extent)

#!=====================================================================================
def assign_dist(dataset_stats, params, center, R):
    (param1, param2) = params
    (x0, y0) = center
    (Rx, Ry) = R
    
    dataset_stats.loc[: , 'distance'] = np.ceil(((dataset_stats[param1] - x0)**2/Rx**2 + (dataset_stats[param2] - y0)**2/Ry**2)**0.5)
        
    return dataset_stats.distance.unique()