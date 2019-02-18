"""
Helper methods for visualizing simulation results
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_incorrect_fraction(reward_histories, plot_legend):
    """
    Plots regret/t vs. t (i.e. fraction of incorrect decisions) with confidence bound
    reward_history: array of reward_history arrays
    plot_legend: legend for the plotted curve (string)
    """
    regret = 0 - np.array(reward_histories)  # value 1 corresponds to incorrect decision
    N, T = regret.shape  # N simulation runs; T steps per run

    cumulative_regret = np.cumsum(regret, axis=1)
    incorrect_fraction = cumulative_regret / np.arange(1, T+1)

    average = np.mean(incorrect_fraction, axis=0)
    upper = np.amax(incorrect_fraction, axis=0)
    lower = np.amin(incorrect_fraction, axis=0)

    t_axis = np.arange(1, T+1)
    plt.fill_between(t_axis, lower, upper, alpha=0.2, color='gray')
    plt.plot(t_axis, average, label=plot_legend)


def plot_validation_accuracy(val_step_history, val_accuracy_histories, plot_legend):
    """
    Plots cross validation accuracy vs. time with confidence bound
    plot_legend: legend for the plotted curve (string)
    """
    accuracy = np.array(val_accuracy_histories)

    average = np.mean(accuracy, axis=0)
    upper = np.amax(accuracy, axis=0)
    lower = np.amin(accuracy, axis=0)

    plt.fill_between(val_step_history, lower, upper, alpha=0.2, color='gray')
    plt.plot(val_step_history, average, label=plot_legend)
