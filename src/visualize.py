"""
Helper methods for visualizing simulation results
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_incorrect_fraction(reward_histories, plot_legend, plot_bound=False, drop_k=0):
    """
    Plots regret/t vs. t (i.e. fraction of incorrect decisions)

    reward_history:
        array of reward_history arrays
    plot_legend:
        legend for the plotted curve (string)
    plot_bound:
        whether to plot the max/min bound
    drop_k:
        drop the largest and smallest 'drop_k' records when averaging and plotting
    """
    regret = 0 - np.array(reward_histories)  # value 1 corresponds to incorrect decision
    N, T = regret.shape  # N simulation runs; T steps per run
    assert N > 2 * drop_k

    cumulative_regret = np.cumsum(regret, axis=1)
    incorrect_fraction = cumulative_regret / np.arange(1, T+1)

    incorrect_fraction.sort(axis=0)
    incorrect_fraction = incorrect_fraction[drop_k:N-drop_k,:]

    average = np.mean(incorrect_fraction, axis=0)
    t_axis = np.arange(1, T+1)

    if plot_bound:
        lower, upper = incorrect_fraction[0,:], incorrect_fraction[-1,:]
        plt.fill_between(t_axis, lower, upper, alpha=0.2, color='gray')

    plt.plot(t_axis, average, label=plot_legend)


def plot_validation_accuracy(val_step_history, val_accuracy_histories, plot_legend, plot_bound=False, drop_k=0):
    """
    Plots cross validation accuracy vs. t

    plot_legend:
        legend for the plotted curve (string)
    plot_bound:
        whether to plot the max/min bound
    drop_k:
        drop the largest and smallest 'drop_k' records when averaging and plotting
    """
    N = len(val_accuracy_histories)
    assert N > 2 * drop_k

    accuracy = np.array(val_accuracy_histories)
    accuracy.sort(axis=0)
    accuracy = accuracy[drop_k:N-drop_k,:]
    average = np.mean(accuracy, axis=0)

    if plot_bound:
        lower, upper = accuracy[0,:], accuracy[-1,:]
        plt.fill_between(val_step_history, lower, upper, alpha=0.2, color='gray')

    plt.plot(val_step_history, average, label=plot_legend)
