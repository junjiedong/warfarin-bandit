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

    def __init__(self, data_file, label_discretizer, test_size=0, add_bias=True,
                       standardize=True, reward_structure="binary"):
        """
        Loads and preprocesses the Warfarin data
        label_discretizer is used to transform continuous dosage to discrete arm labels

        If add_bias is True, a constant bias term is added to the feature matrix
        If standardize is True, feature columns are standardized to have zero mean & unit variance

        Also reserves a hold-out validation set of size 'test_size'

        reward_structure: "binary", "k_level", "quantized_diff"
        """
        # Load data
        data = pd.read_csv(data_file)
        data = preprocess(data, label_discretizer, standardize, add_bias)

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

        assert reward_structure in ("binary", "k_level", "quantized_diff")
        self.reward_structure = reward_structure

        # Average dosage for patients of each dosage-level
        # Used for calculating "k-level" or "quantized_diff" rewards
        self.average_dosage = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            self.average_dosage[i] = np.mean(self.y_continuous_original[self.y_discrete_original == i])

        print("Instantiated a Warfarin Bandit simulator!")
        print("Reward structure:", reward_structure)
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
        decision_correctness_history = []
        val_step_history = []
        val_accuracy_history = []
        confusion_matrix = np.zeros((self.num_arms, self.num_arms))

        for t in range(self.train_size):
            # Choose an arm to pull
            context = self.Xtrain[t,:].reshape((self.num_features, 1))
            arm = policy.choose_arm(context)

            # Evaluate reward for the action
            true_arm = self.ytrain_discrete[t]
            if self.reward_structure == "binary":
                reward = int(arm == true_arm) - 1
            elif self.reward_structure == "k-level":
                reward = -abs(self.average_dosage[arm] - self.average_dosage[true_arm])
            else:
                dosage_diff = abs(self.average_dosage[arm] - self.ytrain_continuous[t])
                reward = -int(dosage_diff)
            reward_history.append(reward)
            decision_correctness_history.append(int(arm == true_arm))

            # Update confusion matrix
            confusion_matrix[true_arm, arm] += 1

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
        self.decision_correctness_history = np.array(decision_correctness_history)
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

    def get_decision_correctness_history(self):
        return self.decision_correctness_history

    def get_number_incorrect_decisions(self):
        return len(self.decision_correctness_history) - np.sum(self.decision_correctness_history)

    def get_total_regret(self):
        # total regret with respect to optimal decisions
        return -self.reward_history.sum()

    def get_confusion_matrix(self):
        return self.confusion_matrix
