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
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

#!==============Define functions================
#!=====================================================================================
def find_tumbles(traj,  
                 model, 
                 params = ['vel_norm', 'acc_norm', 'acc_angle'], 
                n_components = 3, threshold = 1,
                covariance_type = 'diag',
                n_iter = 1000,
                model_type = 'HMM',
                int_state = 1):
    '''
    Finds distinct motility states in bacterial trajectories.
    Takes:
    traj - dataframe, containing bacterial trajectories and their parameters
    model - Gaussian Mixture or Hidden Markov Model trained on a reference dataset, 
            if no model is provided, current dataset is used for training
    params - variable used to infer hidden states
    n_components - number of states
    threshold - percent difference between running velocity calculated from run states, 
                used to indicate convergence of state assignment
    covariance_type - defines whether variables are independent or not, 
                diagonal covariance implies independent variables
    int_state - if '1', intermediate state is assigned to runs, if '0' to tumbles,
                only matters if there are more than 2 components.
    Returns:
    model
    traj - dataframe with assigned states
    '''
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
def assign_tumbles(traj, 
                   vel_type, 
                   state_type, 
                   tbias_type, 
                   int_state):
    '''
    Assign run and tumble states after sequence of hidded states have been established
    Takes:
    traj - dataframe with bacterial trajectories
    vel_type - type of velocity used to establish which state is a run and which is a tumble
    state_type - name of the state, e.g. 'tbias_HMM'
    int_state - if '1', intermediate state is assigned to runs, if '0' to tumbles,
                only matters if there are more than 2 components.
    Returns:
    run_state - number of a hidden state, corresponding to a run
    tumble_state - number of a hidden state, corresponding to a tumble
    vel_mean - average velocity for different states
    '''
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
def fit_model(traj, 
              params, 
              n_components, 
              covariance_type,
              n_iter, 
              model_type,
              model):
    '''
    Estimates model parameters based on the data. Predict states given model parameters and data
    Takes:
    traj - dataframe, containing bacterial trajectories and their parameters
    
    params - names of the columns in the traj, variable used to infer hidden states
    n_components - number of states
    covariance_type - defines whether variables are independent or not, 
                diagonal covariance implies independent variables
    n_iter - maximum number of iterations
    mode_type - 'GMM' (Gaussian mixture), 'HMM' (Hidden Markov Model) or 'GMM_Dirichlet' (Gaussian mixture with Dirichlet emieeions)
    model - Gaussian Mixture or Hidden Markov Model trained on a reference dataset, 
            if no model is provided, current dataset is used for training
    Returns:
    '''
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
def calc_stat(dataset, 
              params, 
              funcs, 
              col_names):
    '''
    Calculates statistics for every trajectory and adds corresponding columns to the dataframe. 
    Takes:
    dataset - dataframe with trajectories
    params - names of the columns in the dataset, parameters for which statistics is calculated
    funcs - functions used to calculate statistics
    col_names - appendices to the new column names
    Returns:
    dataset with calculated statistics
    '''
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
def calc_run_stat(traj, 
                  param, 
                  state):
    '''
    Calculate mean parameters over the frames corresponding to runnint state
    Takes:
    traj - dataframe with trajectories
    param - names of the columns in the traj, parameters for which statistics is calculated
    state - name of the column in the traj,  that is used to define 'run' state
    Returns:
    traj - dataframe with calculated statistics
    '''
    traj.loc[:, param + '_run'] = traj.loc[:, param]
    traj.loc[traj[state] == 0, param + '_run'] = None
    if param + '_run_mean' in traj.keys():
        traj.drop(param + '_run_mean', inplace = True, axis = 1)
        
    traj = calc_stat(traj, [param + '_run'], [np.nanmean], ['mean'])
    return traj
#!=====================================================================================
def calc_params(traj, 
                wind, 
                fps, 
                pix_size):
    '''
    Calculates instantaneous parameters of the trajectories, velocity, angle between consecutive velocity vectors, 
    acceleration, angular acceleration and angular velocity
    Takes:
    traj - dataframe with trajectories
    wind - number of rames in a window used for calculating parameters
    fps - video framerate, number of rames per second
    pix_size - size of the pixel in microns
    Returns:
    '''
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
def calc_diff(traj, 
              param, 
              wind, 
              left_edge, 
              right_edge, 
              idx_edge, 
              dividebywind = False):
    '''
    Calculates change in parameter in a window
    Takes:
    traj - dataframe with trajectories
    param - name of the column in a dataframe, parameter for which the change is calculated
    wind - number of rames in a window used for calculating parameters
    left_edge - number of frames on the left that will be empty
    right_edge -  of frames on the right that will be empty
    idx_edge - frames marking the end of the trajectories
    Returns:
    diff - array of calculated values with the same length as the dataframe provided
    '''
    diff = (traj[param].values[wind:] - traj[param].values[:-wind])
    if dividebywind == True:
        diff = diff/wind
    else:
        diff = diff/(traj['frame'].values[wind:] - traj['frame'].values[:-wind])
    diff = np.concatenate((np.repeat(np.nan, left_edge), diff, np.repeat(np.nan, right_edge)))
    diff[idx_edge] = np.nan
    return diff
#!=====================================================================================
def find_edges(traj, 
               wind):
    '''
    Takes:
    Returns:
    '''
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
    '''
    Takes:
    Returns:
    '''
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
def find_center(ax, 
                X, 
                min_bin_freq = 100):
    '''
    Takes:
    Returns:
    '''
    bandwidth = estimate_bandwidth(X, quantile=0.2)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq = min_bin_freq)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    plt.figure(1)

    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], col + '.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=3, markeredgewidth = 3)
    ax.set_title('Estimated number of clusters: %d' % n_clusters_)
    return cluster_centers, labels

#!=====================================================================================
def find_MADs_MeanShift(ax, 
                        param1, 
                        param2, 
                        data, 
                        min_bin_freq):
    '''
    Takes:
    Returns:
    '''
    x = data[param1].values
    y = data[param2].values
    X = np.column_stack([x, y])
    cluster_centers, labels = find_center(ax, X, min_bin_freq)
    cluster_num = np.where(cluster_centers[:,0] > 10)[0][0]
    (x0, y0) = cluster_centers[cluster_num]
    
    MADx = np.median(abs(x[np.where(labels == cluster_num)] - x0))
    MADy = np.median(abs(y[np.where(labels == cluster_num)] - y0))
    return (MADx, MADy), (x0, y0), cluster_centers, labels


#!=====================================================================================
def plot_MADs((MADx, MADy), 
              (x0, y0), 
              N_MADs, 
              colors, 
              lines):
    '''
    Takes:
    Returns:
    '''
    for i in range(len(N_MADs)):
        ellipse = Ellipse((x0, y0), N_MADs[i]*MADx*2, N_MADs[i]*MADy*2, edgecolor=colors[i], lw=5, ls =lines[i],
                             fc = 'None', label = '{} MAD'.format(N_MADs[i]))
        ax.add_patch(ellipse)

#!=====================================================================================
def assign_dist(dataset_stats, 
                params, 
                center, 
                R):
    '''
    Takes:
    Returns:
    '''
    (param1, param2) = params
    (x0, y0) = center
    (Rx, Ry) = R
    
    dataset_stats.loc[: , 'distance'] = np.ceil(((dataset_stats[param1] - x0)**2/Rx**2 + (dataset_stats[param2] - y0)**2/Ry**2)**0.5)
        
    return dataset_stats.distance.unique()