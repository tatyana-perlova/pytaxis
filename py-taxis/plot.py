#!======Open librairies================
import numpy as np
from pandas import DataFrame, Series  # for convenience
import matplotlib.pyplot as plt
import seaborn
import scipy
from matplotlib.patches import Ellipse

#!==============Define functions================

def traj_len_dist(traj, bw, cutoffs):
    traj_lengths = traj.groupby('particle').count()
    bins = np.arange( traj_lengths.frame.min(), traj_lengths.frame.max(), bw)
    fig = plt.figure(figsize=(4,3.5))
    colors = seaborn.xkcd_palette(['steel', 'jungle green'])

    ax = fig.add_subplot(111)
    n, bins, patches = plt.hist(traj_lengths.frame.values, bins, color = colors[0])
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.9))

    ax.set_ylabel(r'Number of trajectories', color=colors[0])
    ax.tick_params('y', colors=colors[0])

    plt.xlabel('Number of frames/Frame cutoff')
    ax2 = ax.twinx()
    bin_cent = (bins[1:] + bins[:-1])/2
    data_fraction = np.cumsum(n[::-1]*bin_cent[::-1])[::-1]/traj_lengths.frame.sum()
    fraction = scipy.interpolate.interp1d(bin_cent, data_fraction)

    ax2.plot(bin_cent, data_fraction, color = colors[1])
    ax2.fill_between(bin_cent, np.cumsum(n[::-1]*bins[1:][::-1])[::-1]/traj_lengths.frame.sum(), 
                     np.cumsum(n[::-1]*bins[:-1][::-1])[::-1]/traj_lengths.frame.sum(),
                     color = colors[1], alpha = 0.5)
    for cutoff in cutoffs:
        ax2.plot(cutoff, float(fraction(cutoff)), 'ko', alpha = 0.9, markersize = 5)
        ax2.text(cutoff + 20, float(fraction(cutoff)) - 0.01, '({} frames, {}%)'.format(cutoff, round(float(fraction(cutoff))*100) ), fontsize = 13)


    ax2.set_ylabel(r'Fraction of data', color=colors[1])
    ax2.tick_params('y', colors=colors[1])
    ax2.set_ylim(0, 1.05)
    ax.text(0.9, 0.9, r'$<N\,$' + r'$frames> = {}$'.format(int(traj_lengths.frame.mean())), verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize = 14)
    
#!==============KDE================
def plot_KDE(Z, extent, N_traj, vmax = None, tick_step = 1000):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)

    if vmax == None:
        vmax = np.max(Z.flatten())
    aspectratio =1.0*(extent[1] - extent[0])/(1.0*extent[3] - extent[2])
   
    cset = ax.imshow(np.rot90(Z), vmax = vmax, cmap=plt.cm.gist_earth_r, extent=extent, aspect = aspectratio)
    
    tick_labels = np.arange(0, np.ceil(vmax*1000)/1000*N_traj, tick_step)
    ticks = tick_labels/N_traj
    cbar = fig.colorbar(cset, ticks = ticks)
    cbar.ax.set_yticklabels(tick_labels)
    
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    return cset
    
#!==============Ellipses================
def plot_ellipses(ax, (Rx, Ry), (x0, y0), N_Rs, colors):
    for i in range(len(N_Rs)):
        ellipse = Ellipse((x0, y0), N_Rs[i]*Rx*2, N_Rs[i]*Ry*2, edgecolor=colors[i], lw=5,
                             fc = 'None', label = '{} R'.format(N_Rs[i]))
        ax.add_patch(ellipse)
        
#!==============trajectories================
def plot_traj(traj, imdim, palette = None):
    
    if palette == None:
        palette = seaborn.color_palette("Dark2", len(traj.particle.unique()))
    plt.tick_params(\
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',
        left = 'off',
        right = 'off',# ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.yticks([])
    plt.xticks([])
    plt.xlim(0, imdim)
    plt.ylim(0, imdim)
    
    
    unstacked = traj.set_index(['frame', 'particle']).unstack()
    plot = plt.plot(unstacked.x, unstacked.y, linewidth=2, alpha = 1)
    
#!==============distributions================

def dist_by_state(traj, state, params, palette):
    
    plt.figure(figsize = (4*len(params), 3)) 
    for i in range(len(params)):
        bins = np.arange(traj[params[i]].min(), 
                        traj[params[i]].max(), 
                        (traj[params[i]].max() - traj[params[i]].min())/20)
        plt.subplot(1, len(params), i + 1)
        for state_un in traj[state].unique():
            _ = plt.hist(traj[traj[state] == state_un][params[i]].values, 
                            bins = bins, 
                            facecolor = palette[state_un], 
                            alpha = 0.6, label = 'state' + str(state_un))

        _ = plt.hist(traj[params[i]].values, bins = bins, 
                        edgecolor = 'grey', linewidth = 5, histtype='step', label = 'all')
        plt.xlabel(params[i])
        plt.xlim(min(bins), max(bins))
        plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))

    plt.legend(loc = 1, bbox_to_anchor=(1.4, 1.0))


#!=================plot trace===================
def plot_trace(ax, wind, shift, data, fps, color, label = None, line_style = '-', tbias = False):
    aggregations = {'mean': ['mean', 'count'], 'std': lambda x: np.nansum(x**2), 'count':  ['sum', 'mean']}
    data.columns = ['frame', 'mean', 'std', 'count']
    param_stats = data.groupby(['frame'], as_index = False)['mean', 'std', 'count'].agg(aggregations)
    param_stats.loc[:, 'sem'] = (param_stats['std']['<lambda>']/(param_stats['count']['mean']))**0.5/param_stats['mean']['count']
    param_stats.columns = ['_'.join(col).strip() for col in param_stats.columns.values]
    param_stats.drop(['std_<lambda>', 'mean_count', 'count_mean'], inplace = True, axis = 1)
    param_stats.rename(columns={'count_sum': 'N_traj', 'mean_mean': 'param_mean', 'sem_': 'sem'}, inplace=True)
    
    if tbias:
        param_stats['param_mean'] = 1 - param_stats['param_mean']

        
    plot_rolling_mean(ax, param_stats[['frame_', 'param_mean', 'sem']],
                     wind, shift, fps, label, color, line_style, param_names = ('frame_', 'param_mean', 'sem'),
                      )

    return param_stats

#!=================plot rolling mean ===================

def plot_rolling_mean(ax, param_stats, wind, shift, fps, label, color, line_style, 
                      param_names,
                      ):
    frame, param, sem = param_names
    rolling_mean = param_stats.rolling(window = wind, center=True).mean()[::shift]
    ax.plot(rolling_mean[frame]/fps,
            rolling_mean[param],
            color = color,
            label = label,
           linestyle = line_style)
    ax.fill_between(rolling_mean[frame]/fps,
                    rolling_mean[param] - \
                    rolling_mean[sem],
                    rolling_mean[param] + \
                    rolling_mean[sem], 
                    color = color,
                    alpha = 0.3 , label = '')
    
#!=================plot time traces of the trajectory parameters ===================

def plot_params(traj, particle, pix_size, fps, size = 5, params = ['vel', 'acc', 'angle', 'vel_angle', 'acc_angle'],
              colorbystate = False, state = 'tbias_alon', palette = None, alpha = 0.5):
    
    traj_single = traj[traj.particle == particle]
    colors = np.array(seaborn.color_palette("RdBu_r", len(traj_single)))
    if palette == None:
        palette = seaborn.color_palette('Dark2', len(traj_single[state].unique()))
    if colorbystate == True:
        for i in range(len(traj_single[state].unique())):
            state_un = traj_single[state].unique()[i]
            colors[np.where(traj_single[state] == state_un)[0]] = palette[state_un]

    plt.figure(figsize=(size, 3.5 * len(params)))
    
    for i in range(len(params)):
        ax = plt.subplot2grid((len(params), 4), (i, 0), colspan = 3)
        ax.plot(traj_single['frame'].values/fps, traj_single[params[i]].values, '--', color = 'grey', alpha = 0.5)
        ax.scatter(traj_single['frame'].values/fps, traj_single[params[i]].values, c=colors, s = 35)
        plt.ylabel(params[i])
        plt.xlim(traj_single['frame'].min()/fps, traj_single['frame'].max()/fps)

        ax1 = plt.subplot2grid((len(params), 4), (i, 3))
        ax1.hist(traj_single[params[i]].values[~np.isnan(traj_single[params[i]].values)], bins = 20,
                 range = ax.get_ylim(),
                 normed = 1,
                 orientation = 'horizontal', 
                color = 'grey')

        ax1.set_yticks([])
        ax1.set_xticks([])
    ax.set_xlabel('Time, s')
    
#!=================plot individual bacterial trajectories ===================

def plot_traj(traj, particles, pix_size, size = 5, colorbystate = False, state = 'tbias_alon', palette = None, sizes = None, alpha = 0.5):
    if palette == None:
        palette = seaborn.color_palette('Dark2', len(traj_single[state].unique()))
    if sizes == None:
        sizes = np.repeat(size*3, len(traj[state].unique()))

    for particle in particles:
        traj_single = traj[traj.particle == particle]        
        fig1 = plt.figure(figsize=(size,size))
        plt.plot(traj_single.x.values*pix_size, traj_single.y.values*pix_size, '--', color = 'grey', alpha = 0.5)
        for i in np.sort(traj_single[state].unique())[::-1]:
            plt.scatter(traj_single[traj_single[state] == i].x.values*pix_size, 
                        traj_single[traj_single[state] == i].y.values*pix_size, 
                        c=palette[i], 
                        s = sizes[i], 
                        alpha = 1, 
                        edgecolor = palette[i])
    plt.xlim(traj_single.x.min()*pix_size - 5, traj_single.x.max()*pix_size + 5)
    plt.ylim(traj_single.y.min()*pix_size - 5, traj_single.y.max()*pix_size + 5)

    plt.xlabel(r'x, $\mu m$')
    plt.ylabel(r'y, $\mu m$')
    plt.gca().set_aspect(1)