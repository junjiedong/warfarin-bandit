import numpy as np


class Policy(object):
    """
    Abstract class for a contextual multi-armed bandit policy/algorithm
    """

    def __init__(self):
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


class FixedDosePolicy(Policy):
    """
    FixedDosePolicy doesn't learn anything
    It chooses arm 1 (i.e. medium dose) all the time
    """

    def __init__(self):
        pass

    def choose_arm(self, context, eval=False):
        return 1  # medium

    def update_policy(self, context, arm, reward):
        pass


class EpsilonGreedyPolicy(Policy):
    """
    At each step, this policy either chooses a random action with probability 'eps',
    or chooses the optimal action based on current reward estimates

    The parameters of each arm are fitted using ordinary least-squares

    The implementation is not as efficient as it could be, just a quick prototype :)
    """

    def __init__(self, context_dim, num_arms, eps=0.2):
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.eps = eps
        self.step = 0

        # linear regression parameters
        self.beta = np.zeros((num_arms, context_dim))
        self.X_history = [None] * num_arms
        self.y_history = [None] * num_arms

    def choose_arm(self, context, eval=False):
        if eval or np.random.binomial(1, self.eps) == 0:
            # Choose optimal action based on current reward estimates
            scores = np.dot(self.beta, context).reshape((self.num_arms,))
            return np.argmax(scores)
        else:
            # Uniformly sample an action
            return np.random.choice(self.num_arms)

    def update_policy(self, context, arm, reward):
        self.step += 1

        # Update history
        context = context.T.copy()
        if self.X_history[arm] is None:
            self.X_history[arm] = context
            self.y_history[arm] = np.array([[reward]])
        else:
            self.X_history[arm] = np.vstack((self.X_history[arm], context))
            self.y_history[arm] = np.vstack((self.y_history[arm], np.array([[reward]])))

        # Update parameters
        X, y = self.X_history[arm], self.y_history[arm]
        self.beta[arm,:] = np.dot(np.linalg.pinv(X), y).reshape((self.context_dim,))
