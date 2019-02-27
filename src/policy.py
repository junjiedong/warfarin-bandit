"""
Policies for general contextual multi-armed bandit:
1. LinUCBPolicy
2. LinearThompsonPolicy
3. EpsilonGreedyPolicy
4. UCB1Policy (simply ignores the context)

Policies for Warfarin bandit:
1. WarfarinFixedDosePolicy
2. WarfarinLinearOraclePolicy
3. WarfarinLogisticOraclePolicy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from data_proc import *


class Policy(object):
    """
    Abstract class for a contextual multi-armed bandit policy/algorithm
    """

    def __init__(self):
        raise NotImplementedError

    def reset(self):
        """
        Resets the policy to its initial states
        """
        raise NotImplementedError

    def choose_arm(self, context, eval=False):
        """
        Returns the arm to pull based on 'context'
        context: Numpy matrix of shape (context_dim, 1)

        If eval is True, the policy should always act greedily,
        otherwise, exploration strategy is applied
        """
        raise NotImplementedError

    def update_policy(self, context, arm, reward):
        """
        Updates the parameters of the policy based on feedback from a single step
        context: Numpy matrix of shape (context_dim, 1)
        """
        raise NotImplementedError


class WarfarinFixedDosePolicy(Policy):
    """
    WarfarinFixedDosePolicy doesn't learn anything
    It chooses one fixed arm all the time
    """

    def __init__(self, arm):
        self.arm = arm

    def reset(self):
        pass

    def choose_arm(self, context, eval=False):
        return self.arm

    def update_policy(self, context, arm, reward):
        pass


class WarfarinLinearOraclePolicy(Policy):
    """
    WarfarinLinearOraclePolicy pre-trains a linear regression estimator for each arm using all data
    No online policy based on linear regression can beat the performance of this policy
    """

    def __init__(self, data_file, label_discretizer):
        """
        data_file: path to raw Warfarin dataset file
        label_discretizer: function that maps continuous daily dosage to 0 ~ K-1
        """
        # Prepare training data
        data = pd.read_csv(data_file)
        data = preprocess(data, label_discretizer)
        data['bias'] = 1.0  # manually add a bias term

        self.num_arms = data['dosage-level'].nunique()  # number of dosage levels (i.e. arms/classes)
        X = data.drop(['daily-dosage', 'dosage-level'], axis=1).values
        y = [(data['dosage-level'] == l).astype(np.float32).values for l in range(self.num_arms)]
        self.num_features = X.shape[1]

        # Train K linear regression estimators with no regularization
        self.theta = np.zeros((self.num_arms, self.num_features)) # linear estimators
        for i in range(self.num_arms):
            self.theta[i,:] = np.dot(np.linalg.pinv(X), y[i]).reshape((self.num_features,))

    def reset(self):
        # Reset has no effect since the internal models are trained using all data
        pass

    def choose_arm(self, context, eval=False):
        scores = np.dot(self.theta, context).reshape((self.num_arms,))
        return np.argmax(scores)

    def update_policy(self, context, arm, reward):
        pass


class WarfarinLogisticOraclePolicy(Policy):
    """
    WarfarinLogisticOraclePolicy pre-trains a Logistic classifier for each arm using all data
    This policy only works when the bandit rewards are binary
    """

    def __init__(self, data_file, label_discretizer):
        """
        data_file: path to raw Warfarin dataset file
        label_discretizer: function that maps continuous daily dosage to 0 ~ K-1
        """
        # Prepare training data
        data = pd.read_csv(data_file)
        data = preprocess(data, label_discretizer)
        data['bias'] = 1.0  # manually add a bias term

        K = data['dosage-level'].nunique()  # number of dosage levels (i.e. arms/classes)
        X = data.drop(['daily-dosage', 'dosage-level'], axis=1).values
        y = [(data['dosage-level'] == l).astype(np.float32).values for l in range(K)]

        # Train K Logistic Classifiers for the K arms (large C -> no regularization)
        self.models = [LogisticRegression(C=100000, fit_intercept=False, solver='liblinear') for _ in range(K)]
        for i in range(K):
            self.models[i].fit(X, y[i])

    def reset(self):
        # Reset has no effect since the internal models are trained using all data
        pass

    def choose_arm(self, context, eval=False):
        scores = [m.predict_proba(context.reshape((1, -1)))[0,1] for m in self.models]
        return np.argmax(scores)

    def update_policy(self, context, arm, reward):
        pass

    def get_estimators(self):
        return self.models


class LinUCBPolicy(Policy):
    """
    LinUCBPolicy implements the LinUCB algorithm in paper "A Contextual Bandit Approach
    to Personalized News Article Recommendation".
    It fits a ridge regression estimator for each arm, and uses the idea of upper confidence
    bound to balance exploration and exploitation.

    There is a single tuning parameter 'alpha' that controls the exploration tradeoff
    """

    def __init__(self, context_dim, num_arms, alpha=1.0):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.step = 0
        self.num_selected = np.zeros(self.num_arms) # number of times each arm is selected
        self.num_suboptimal_actions = 0 # number of times we select suboptimal actions for exploration

        self.theta = np.zeros((self.num_arms, self.context_dim)) # linear estimators
        self.A = [np.eye(self.context_dim) for _ in range(self.num_arms)]
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

    def choose_arm(self, context, eval=False):
        reward_estimates = np.dot(self.theta, context).reshape((self.num_arms,))
        greedy_arm = np.argmax(reward_estimates)
        if eval:
            return greedy_arm  # choose greedily
        else:
            confidence_bonus = np.zeros(self.num_arms)
            for i in range(self.num_arms):
                confidence_bonus[i] = self.alpha * np.sqrt(context.T.dot(self.A_inv[i]).dot(context)[0,0])

            select_arm = np.argmax(reward_estimates + confidence_bonus)
            if greedy_arm != select_arm:
                self.num_suboptimal_actions += 1
            return select_arm

    def update_policy(self, context, arm, reward):
        self.step += 1
        self.num_selected[arm] += 1

        # update linear estimators
        self.A[arm] += np.dot(context, context.T)
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] += reward * context
        self.theta[arm,:] = np.dot(self.A_inv[arm], self.b[arm]).reshape((self.context_dim,))

    def get_action_counts(self):
        return self.num_selected

    def get_num_suboptimal_actions(self):
        # Number of times we select suboptimal actions for exploration
        return self.num_suboptimal_actions


class LinearThompsonPolicy(Policy):
    """
    Algorithm "Thompson Sampling for contextual bandits with linear payoffs" (ICML13)

    A single hyperparameter v controls the exploration/exploitation tradeoff
    larger v -> higher variance for prior distribution -> more exploration
    """

    def __init__(self, context_dim, num_arms, v=0.01):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.v = v  # scaling factor of covariance matrix; controls exploration tradeoff
        self.reset()

    def reset(self):
        self.step = 0
        self.num_selected = np.zeros(self.num_arms) # number of times each arm is selected

        self.theta = np.zeros((self.num_arms, self.context_dim)) # linear estimators
        self.A = [np.eye(self.context_dim) for _ in range(self.num_arms)]
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

    def choose_arm(self, context, eval=False):
        if eval:
            return np.argmax(np.dot(self.theta, context).squeeze())
        else:
            scores = np.zeros(self.num_arms)
            for i in range(self.num_arms):
                # Sample model parameter from prior distribution
                theta_sample = np.random.multivariate_normal(mean=self.theta[i,:],
                                                             cov=(self.v * self.A_inv[i]))
                # Calculate expected payoff
                scores[i] = np.sum(theta_sample * context.squeeze())
            return np.argmax(scores)

    def update_policy(self, context, arm, reward):
        self.step += 1
        self.num_selected[arm] += 1

        # compute new posterior distribution
        self.A[arm] += np.dot(context, context.T)
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] += reward * context
        self.theta[arm,:] = np.dot(self.A_inv[arm], self.b[arm]).reshape((self.context_dim,))

    def get_action_counts(self):
        return self.num_selected


class EpsilonGreedyPolicy(Policy):
    """
    At step t, this policy either chooses a random action with probability
    eps_schedule(t), or chooses the optimal action based on current reward estimates.
    When eps_schedule(t) is always zero, the policy becomes a purely greedy policy

    The parameters of each arm are fitted using ordinary least-squares
    """

    def __init__(self, context_dim, num_arms, eps_schedule=lambda t: 0.2):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.eps_schedule = eps_schedule
        self.reset()

    def reset(self):
        self.step = 0
        self.theta = np.zeros((self.num_arms, self.context_dim)) # linear estimators
        self.A = [np.eye(self.context_dim) for _ in range(self.num_arms)]
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

    def choose_arm(self, context, eval=False):
        eps = max(self.eps_schedule(self.step + 1), 0)
        if eval or np.random.binomial(1, eps) == 0:
            # Choose optimal action based on current reward estimates
            reward_estimates = np.dot(self.theta, context).reshape((self.num_arms,))
            return np.argmax(reward_estimates)
        else:
            # Uniformly sample an action
            return np.random.choice(self.num_arms)

    def update_policy(self, context, arm, reward):
        self.step += 1

        # update linear estimators
        self.A[arm] += np.dot(context, context.T)
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] += reward * context
        self.theta[arm,:] = np.dot(self.A_inv[arm], self.b[arm]).reshape((self.context_dim,))


class UCB1Policy(Policy):
    """
    Upper Confidence Bound Algorithm (UCB1) for non-contextual multi-armed bandit
    For contextual bandit, UCB1 will simply ignore the context
    """

    def __init__(self, num_arms):
        self.num_arms = num_arms

    def reset(self):
        self.step = 0
        self.num_selected = np.zeros(self.num_arms) # number of times each arm is selected
        self.sum_of_rewards = np.zeros(self.num_arms) # cumulative rewards for each arm
        self.reward_estimates = np.zeros(self.num_arms)

    def choose_arm(self, context, eval=False):
        if eval:
            return self.get_optimal_arm()
        else:
            scores = np.zeros(self.num_arms)
            bonus_constant = 2 * np.log(self.step + 1)
            for i in range(self.num_arms):
                if self.num_selected[i] == 0:
                    scores[i] = 1000000
                else:
                    scores[i] = self.reward_estimates[i] + np.sqrt(bonus_constant / self.num_selected[i])
            return np.argmax(scores)

    def update_policy(self, context, arm, reward):
        self.step += 1
        self.num_selected[arm] += 1
        self.sum_of_rewards[arm] += reward
        self.reward_estimates[arm] = self.sum_of_rewards[arm] / self.num_selected[arm]

    def get_optimal_arm(self):
        return np.argmax(self.reward_estimates)

    def get_reward_estimates(self):
        return self.reward_estimates

    def get_action_counts(self):
        return self.num_selected


class LinUCBSafePolicy(Policy):
    """
    Exactly the same as LinUCB, except we don't allow risky exploration:
        - Explore high when greedy action is low, OR
        - Explore low when greedy action is high
    Q: Is this cheating?
    """
    def __init__(self, context_dim, num_arms, alpha=1.0):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.step = 0
        self.num_selected = np.zeros(self.num_arms) # number of times each arm is selected
        self.num_suboptimal_actions = 0 # number of times we select suboptimal actions for exploration

        self.theta = np.zeros((self.num_arms, self.context_dim)) # linear estimators
        self.A = [np.eye(self.context_dim) for _ in range(self.num_arms)]
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

    def choose_arm(self, context, eval=False):
        reward_estimates = np.dot(self.theta, context).reshape((self.num_arms,))
        greedy_arm = np.argmax(reward_estimates)
        if eval:
            return greedy_arm  # choose greedily
        else:
            confidence_bonus = np.zeros(self.num_arms)
            for i in range(self.num_arms):
                confidence_bonus[i] = self.alpha * np.sqrt(context.T.dot(self.A_inv[i]).dot(context)[0,0])

            scores = reward_estimates + confidence_bonus
            select_arm = np.argmax(scores)

            # Prevent risky exploration
            if select_arm == 0 and greedy_arm == 2:
                select_arm = 1 if scores[1] >= scores[2] else 2
            if select_arm == 2 and greedy_arm == 0:
                select_arm = 1 if scores[1] >= scores[0] else 0

            if greedy_arm != select_arm:
                self.num_suboptimal_actions += 1
            return select_arm

    def update_policy(self, context, arm, reward):
        self.step += 1
        self.num_selected[arm] += 1

        # update linear estimators
        self.A[arm] += np.dot(context, context.T)
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] += reward * context
        self.theta[arm,:] = np.dot(self.A_inv[arm], self.b[arm]).reshape((self.context_dim,))

    def get_action_counts(self):
        return self.num_selected

    def get_num_suboptimal_actions(self):
        # Number of times we select suboptimal actions for exploration
        return self.num_suboptimal_actions
