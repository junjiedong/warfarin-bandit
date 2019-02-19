import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from util import Progbar
from data_proc import *


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
        self.X_original = data.drop(['daily-dosage', 'dosage-level'], axis=1).values
        self.y_original = data['dosage-level'].map({'low': 0, 'medium': 1, 'high': 2}).values

        self.test_size = test_size
        self.train_size = self.X_original.shape[0] - test_size
        self.p = self.X_original.shape[1]  # number of features

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


    def reshuffle(self, random_seed=None):
        """
        Re-shuffles the data to generate a new permutation & train/test split
        """
        self.X, self.y = shuffle(self.X_original, self.y_original, random_state=random_seed)
        self.Xtrain, self.ytrain = self.X[self.test_size:,:], self.y[self.test_size:]
        if self.test_size != 0:
            self.Xtest, self.ytest = self.X[:self.test_size,:], self.y[:self.test_size]


    def simulate(self, policy, eval_every=50, random_seed=None):
        """
        Iterates through the entire training set for online policy learning & evaluation
        Performs cross validation every 'eval_every' iterations if test_size > 0

        Every time this method is called, the dataset is first re-shuffled using the given
        random_seed. This makes simulating a policy multiple times more convenient
        """
        self.reshuffle(random_seed)

        reward_history = []  # rewards received at every step
        val_step_history = []  # steps at which cross validation is performed
        val_accuracy_history = []  # accuracies on validation set

        progbar = Progbar(target=self.train_size)  # progress bar
        for t in range(self.train_size):
            # Choose an arm to pull
            context = self.Xtrain[t,:].reshape((self.p, 1))
            arm = policy.choose_arm(context)

            # Evaluate reward for the action
            reward = int(arm == self.ytrain[t]) - 1
            reward_history.append(reward)

            # Update policy based on reward feedback
            policy.update_policy(context, arm, reward)

            # Periodically evaluate accuracy on hold-out validation set
            if self.test_size != 0 and (t % eval_every == 0 or t == (self.train_size-1)):
                correct_count = 0
                for i in range(self.test_size):
                    c = self.Xtest[i,:].reshape((self.p, 1))
                    correct_count += (policy.choose_arm(c, eval=True) == self.ytest[i])
                val_accuracy_history.append(correct_count / self.test_size)
                val_step_history.append(t)

            # Periodically update the progress bar
            if t % 50 == 0 or t == (self.train_size-1):
                progbar.update(t+1)

        self.reward_history = np.array(reward_history)
        if self.test_size != 0:
            self.val_step_history = np.array(val_step_history)
            self.val_accuracy_history = np.array(val_accuracy_history)


    """
    The following getters return the results of the most recent simulation
    """
    def get_validation_history(self):
        return (self.val_step_history, self.val_accuracy_history)

    def get_reward_history(self):
        return self.reward_history

    def get_total_regret(self):
        # total regret with respect to optimal decisions
        return -self.reward_history.sum()
