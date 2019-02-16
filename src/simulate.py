import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle

from data_proc import *
from policy import *


class WarfarinSimulator():
    """
    Simulates online learning of contextual bandit policies on the Warfarin dataset
    """

    def __init__(self, data_file, test_size=0, add_bias=True):
        """
        Loads, preprocesses, and shuffles the Warfarin data
        If add_bias is True, a constant bias term is added to the feature matrix

        Also reserves a hold-out validation set of size 'test_size'
        """
        # Load data
        data = pd.read_csv(data_file)
        data = preprocess(data)
        if add_bias:
            data['bias'] = 1.0
        self.X = data.drop(['daily-dosage', 'dosage-level'], axis=1).values
        self.y = data['dosage-level'].map({'low': 0, 'medium': 1, 'high': 2}).values
        self.test_size = test_size
        self.train_size = self.X.shape[0] - test_size
        self.p = self.X.shape[1]  # number of features

        # Randomly shuffle the data
        self.reshuffle()

        # Store most recent simulation results
        self.reward_history = None
        self.val_accuracy_history = None

        # Logging
        print("Instantiated a Warfarin Bandit simulator!")
        print("Number of features: {}".format(self.p))
        print("Size of training set for online learning: {}".format(self.train_size))
        print("Size of holdout validation set: {}".format(test_size))

    def reshuffle(self):
        """
        Re-shuffles the data to generate a new permutation & train/test split
        """
        self.X, self.y = shuffle(self.X, self.y)
        self.Xtrain, self.ytrain = self.X[self.test_size:,:], self.y[self.test_size:]
        if self.test_size != 0:
            self.Xtest, self.ytest = self.X[:self.test_size,:], self.y[:self.test_size]

    def simulate(self, policy):
        """
        Iterates through the entire training set for online policy learning & evaluation
        Performs cross validation after every iteration if test_size > 0

        Every time this method is called, the dataset is re-shuffled first
        This makes simulating a policy multiple times more convenient
        """
        self.reshuffle()

        reward_history = []
        val_accuracy_history = []

        print("Start simulation...")
        for t in tqdm(range(self.train_size)):
            # Choose an arm to pull
            context = self.Xtrain[t,:].reshape((self.p, 1))
            arm = policy.choose_arm(context)

            # Evaluate reward for the action
            reward = int(arm == self.ytrain[t]) - 1
            reward_history.append(reward)

            # Update policy based on reward feedback
            policy.update_policy(context, arm, reward)

            # Evaluate accuracy on hold-out validation set
            if self.test_size != 0:
                correct_count = 0
                for i in range(self.test_size):
                    c = self.Xtest[i,:].reshape((self.p, 1))
                    correct_count += (policy.choose_arm(c, eval=True) == self.ytest[i])
                val_accuracy_history.append(correct_count / self.test_size)

        self.reward_history = np.array(reward_history)
        if self.test_size != 0:
            self.val_accuracy_history = np.array(val_accuracy_history)

    def get_context_dim(self):
        return self.p

    """
    The following getters return the results of the most recent simulation
    """
    def get_val_accuracy_history(self):
        return self.val_accuracy_history

    def get_reward_history(self):
        return self.reward_history

    def get_total_regret(self):
        return -self.reward_history.sum()


if __name__ == "__main__":
    # Simulate with a size-500 hold-out validation set
    simulator = WarfarinSimulator("../data/warfarin.csv", test_size=500, add_bias=True)

    # Simulate the Fixed Dose Policy
    fixed_policy = FixedDosePolicy()
    simulator.simulate(fixed_policy)
    print("FixedDose total regret: {}".format(simulator.get_total_regret()))
    # Simulate again (data is automatically re-shuffled)
    simulator.simulate(fixed_policy)
    print("FixedDose total regret: {}".format(simulator.get_total_regret()))

    # Simulate the eGreedy policy
    egreedy_policy = EpsilonGreedyPolicy(simulator.get_context_dim(), num_arms=3, eps=0.2)
    simulator.simulate(egreedy_policy)
    print("Epsilon-greedy total regret: {}".format(simulator.get_total_regret()))
    holdout_accuracy = simulator.get_val_accuracy_history()
    plt.plot(np.arange(holdout_accuracy.shape[0]), holdout_accuracy)
    plt.show()
