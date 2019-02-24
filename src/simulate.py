import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from util import Progbar
from data_proc import *


class WarfarinSimulator():
    """
    Simulates online learning of contextual bandit policies on the Warfarin dataset

    Any label_discretizer function can be passed in upon initialization to specify how
    continuous dosage labels are transformed to discrete arm labels
    """

    def __init__(self, data_file, label_discretizer, test_size=0, add_bias=True):
        """
        Loads and preprocesses the Warfarin data
        label_discretizer is used to transform continuous dosage to discrete arm labels

        If add_bias is True, a constant bias term is added to the feature matrix

        Also reserves a hold-out validation set of size 'test_size'
        """
        # Load data
        data = pd.read_csv(data_file)
        data = preprocess(data, label_discretizer)
        if add_bias:
            data['bias'] = 1.0

        self.X_original = data.drop(['daily-dosage', 'dosage-level'], axis=1).values
        self.y_continuous_original = data['daily-dosage'].values  # real-valued label
        self.y_discrete_original = data['dosage-level'].values  # discretized label
        self.test_size = test_size
        self.train_size = self.X_original.shape[0] - test_size
        self.num_features = self.X_original.shape[1]  # number of features
        self.num_arms = data['dosage-level'].nunique()  # number of arms
        self.reward_history = None  # rewards received at every step
        self.val_step_history = None  # steps at which cross validation is performed
        self.val_accuracy_history = None  # accuracies on validation set
        self.confusion_matrix = None

        print("Instantiated a Warfarin Bandit simulator!")
        print("Number of arms: {}".format(self.num_arms))
        print("Number of features: {}".format(self.num_features))
        print("Size of training set for online learning: {}".format(self.train_size))
        print("Size of holdout validation set: {}".format(test_size))


    def reshuffle(self, random_seed=None):
        """
        Re-shuffles the data to generate a new permutation & train/test split
        """
        self.X, self.y_continuous, self.y_discrete = shuffle(self.X_original,
                                                             self.y_continuous_original,
                                                             self.y_discrete_original,
                                                             random_state=random_seed)
        self.Xtrain = self.X[self.test_size:,:]
        self.ytrain_continuous = self.y_continuous[self.test_size:]
        self.ytrain_discrete = self.y_discrete[self.test_size:]
        if self.test_size != 0:
            self.Xtest= self.X[:self.test_size,:]
            self.ytest_continuous = self.y_continuous[:self.test_size]
            self.ytest_discrete = self.y_discrete[:self.test_size]


    def simulate(self, policy, eval_every=50, random_seed=None):
        """
        Iterates through the entire training set for online policy learning & evaluation
        Performs cross validation every 'eval_every' iterations if test_size > 0

        Every time this method is called, the dataset is first re-shuffled using the given
        random_seed. This makes simulating a policy multiple times more convenient
        """
        self.reshuffle(random_seed)

        reward_history = []
        val_step_history = []
        val_accuracy_history = []
        confusion_matrix = np.zeros((self.num_arms, self.num_arms))

        for t in range(self.train_size):
            # Choose an arm to pull
            context = self.Xtrain[t,:].reshape((self.num_features, 1))
            arm = policy.choose_arm(context)

            # Evaluate reward for the action
            """
            NOTE: For real-valued reward, calculate 'reward' using self.ytrain_continuous.
                  If also need to calculate the average dosage level for each arm,
                  do it in the __init__ method so that the code works for any
                  dosage label discretizer functions (do not hardcode the values!)

                  For binary reward, regret history can be easily inferred from
                  'reward_history'. If we want to evaluate regret for real-valued
                  reward, then record the regret in a 'regret_history' array
            """
            reward = int(arm == self.ytrain_discrete[t]) - 1  # binary reward
            reward_history.append(reward)

            # Update confusion matrix
            confusion_matrix[self.ytrain_discrete[t], arm] += 1

            # Update policy based on reward feedback
            policy.update_policy(context, arm, reward)

            # Periodically evaluate accuracy on hold-out validation set
            if self.test_size != 0 and (t % eval_every == 0 or t == (self.train_size-1)):
                correct_count = 0
                for i in range(self.test_size):
                    c = self.Xtest[i,:].reshape((self.num_features, 1))
                    correct_count += (policy.choose_arm(c, eval=True) == self.ytest_discrete[i])
                val_accuracy_history.append(correct_count / self.test_size)
                val_step_history.append(t)

        self.reward_history = np.array(reward_history)
        self.confusion_matrix = confusion_matrix
        self.val_step_history = np.array(val_step_history) if self.test_size != 0 else None
        self.val_accuracy_history = np.array(val_accuracy_history) if self.test_size != 0 else None


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

    def get_confusion_matrix(self):
        return self.confusion_matrix
