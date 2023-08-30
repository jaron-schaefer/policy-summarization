"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)
"""
from itertools import product
import numpy as np
import numpy.random as rn
from models.value_iteration import *
from threading import Thread, Lock

class Maxent:

    def __init__(self, discount=0.99, epochs=300, learning_rate=3e-1, gd_threshold=1e-3, mc_rollouts=0, rollout_horizon=20, par=True):
        """
        Initializes a Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) object.

        :param discount: The discount factor for the MDP. (default=0.90)
        :type discount: float
        :param epochs: The number of gradient descent (GD) epochs to run. (default=500)
        :type epochs: int
        :param learning_rate: The learning rate to use for GD. (default=3e-1)
        :type learning_rate: float
        :param gd_threshold: The convergence threshold for GD. (default=1e-3)
        :type gd_threshold: float
        :param mc_rollouts: The number of Monte Carlo (MC) rollouts to use for expected state visitation frequency (SVF)
                            approximation. Set to 0 for direct calculation without MC. (default=0)
        :type mc_rollouts: int
        :param rollout_horizon: The horizon to use for MC rollouts. (default=20)
        :type rollout_horizon: int
        :param par: Whether to run gradient descent in parallel using multiple threads. (default=True)
        :type par: bool

        This constructor initializes a MaxEnt IRL object with the specified MDP and MaxEnt IRL parameters. The object can be used
        to calculate the reward function for the MDP based on given trajectories using the MaxEnt IRL algorithm.
        """
        self.discount = discount
        self.epochs = epochs 
        self.learning_rate = learning_rate 
        self.gd_threshold = gd_threshold

        ##Expected svf approximation parms
        self.mc_rollouts = mc_rollouts  ##Number of MC rollouts (set to 0 for direct calculatio without MC)
        self.rollout_horizon = rollout_horizon

        ##Multithread or not
        self.par = par

    def irl(self, n_actions, transition_probability, feature_matrix, trajectories, terminal_states):
        """
        Solves for the reward function of an MDP given a set of trajectories.

        :param n_actions: The number of possible actions in the MDP.
        :type n_actions: int
        :param transition_probability: A 3D numpy array representing the transition probability matrix of the MDP. The
            first dimension is the current state, the second dimension is the action taken, and the third dimension is
            the resulting state.
        :type transition_probability: numpy.ndarray
        :param feature_matrix: A 2D numpy array representing the feature matrix of the MDP. The rows correspond to the states
            and the columns correspond to the features.
        :type feature_matrix: numpy.ndarray
        :param trajectories: A list of trajectories. Each trajectory is a list of tuples representing (state, action)
            pairs. Those are the summary trajectories on which the policy is reconstructed.
        :type trajectories: list[list[tuple[int, int]]]
        :param terminal_states: A list of terminal states in the MDP.
        :type terminal_states: list[int]
        :return: A 1D numpy array representing the reward function of the MDP, a 2D numpy array representing the Q-values
            of the MDP, and a 1D numpy array representing the feature counts of the MDP.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        This method uses inverse reinforcement learning to solve for the reward function of an MDP given a set of
        trajectories. It first sets the MDP parameters, including the number of actions, transition probability matrix,
        feature matrix, and terminal states. It then calculates the feature expectations and start state frequencies based on
        the input trajectories. It initializes the weight vector randomly and uses gradient descent to iteratively update the
        weight vector. At each iteration, it calculates the Q-values and the expected state visitation frequencies using the
        current reward vector, and then calculates the gradient of the negative log-likelihood of the trajectories with respect
        to the reward vector. The gradient is used to update the weight vector. The method terminates when the change in the
        reward vector is below the specified threshold or the maximum number of epochs is reached. Finally, it returns the
        learned reward vector, Q-values, and feature counts.
        """

        ##MDP state and action data
        self.n_actions = n_actions
        self.actions = range(n_actions)
        self.transition_probability = transition_probability
        self.feature_matrix = feature_matrix
        self.n_states, self.d_states = feature_matrix.shape # d.states = 
        self.terminal_states = terminal_states

        ##Trajectories from original policy
        self.trajectories = trajectories

        ##Initialise weights.
        alpha = rn.uniform(size=(self.d_states,))

        ##Calculate the feature expectations \tilde{phi}
        feature_expectations, norm_feature_expectations, feature_counts_by_trajectory = self.find_feature_expectations()

        #print(norm_feature_expectations)
        ##Calculate start state frequencies based on trajectories
        p_start_state = self.find_start_state_freq()

        grads = [] ##Plot gradients
        ##Gradient descent on alpha
        diff = 0
        ##Run for epoch iterations or util diff<gd_threshold
        for i in range(self.epochs):
            

            r = self.feature_matrix.dot(alpha)

            ##Check changes in reward vector
            if i > 0:
                diff = np.max(abs(last_r - r))
            if diff <= self.gd_threshold and i > 0:
                ##No significant changes - finish gd
                break
                #pass

            else:
                ##Significant changes in reward vector - calculate another iteration

                if self.mc_rollouts > 0:
                    ##Approximate expected svf with MC Rollouts
                    expected_svf, q_values = self.find_expected_svf_MC(r, p_start_state)
                else:
                    ##Calculate expected svf directly for all state action pairs
                    expected_svf, q_values = self.find_expected_svf(r, p_start_state)

                ##Normalize expected svf to sum to 1
                norm_expected_svf = expected_svf / expected_svf.sum()
                grad = norm_feature_expectations - self.feature_matrix.T.dot(norm_expected_svf) #eq. 6 in paper

                alpha += ( self.learning_rate * grad )

                ##Save reward vector from last iteration
                last_r = np.copy(r)

                grads.append( np.abs( grad ).sum() )
                                                                                # alpha == reward weights
        return self.feature_matrix.dot(alpha).reshape((self.n_states,)), q_values, alpha #self.feature_matrix.T.dot(norm_expected_svf)

    def find_feature_expectations(self):     
        """
        Calculates the feature expectations for given trajectories and returns the average path feature vector.
        
        :return: A tuple of the following:
                - avg_feature_expectations (np.ndarray): Average feature expectations over all trajectories
                - norm_feature_expectations (np.ndarray): Normalized feature expectations over all trajectories
                - feature_counts_by_trajectory (np.ndarray): Sum of feature counts for each trajectory
        """

        ##array of feature expectations- each entry represents a feature
        sum_feature_expectations = np.zeros(self.feature_matrix.shape[1])

        feature_counts_by_trajectory = [] ##Feature counts sum for each trajectory
        n_steps = 0

        for trajectory in self.trajectories:
            trajectory_feature_counts = np.zeros(self.feature_matrix.shape[1])
            for state, _ in trajectory:
                sum_feature_expectations += self.feature_matrix[state]
                trajectory_feature_counts += self.feature_matrix[state]
                n_steps += 1

            feature_counts_by_trajectory.append(trajectory_feature_counts)

        ##Average trajectory feature expectations - divide sum by num of trajectories (original definition in paper)
        avg_feature_expectations =  sum_feature_expectations / len(self.trajectories)

        ##Normalized feature expectations - divide sum by total num of steps in trajectories
        norm_feature_expectations = sum_feature_expectations / n_steps

        return avg_feature_expectations, norm_feature_expectations, np.array(feature_counts_by_trajectory)

    def find_expected_svf_MC(self, r, p_start_state):
        """    
        Calculates expected visitation frequencies for all states with MC simulations based on current policy.

        :param r: reward vector
        :type r: numpy array
        :param p_start_state: probability of starting from each state
        :type p_start_state: numpy array
        :return: expected state visitation frequencies, policy
        :rtype: tuple of two numpy arrays

        This method calculates the probability for each action and state for a given policy, initializes the 
        state visitation frequency (svf) matrix, and then runs Monte Carlo (MC) rollouts to calculate the expected 
        svf for all states. The MC rollouts can be run in parallel or serially, depending on the `par` attribute. 
        """

        ##Calculates probability for each action and state - stochastic policy (lines 1-3 alg 1)
        policy = find_stochastic_policy(self.n_states, self.n_actions,self.transition_probability, r, self.terminal_states,
                             self.discount)

        ##Initialize svf matrix and lock
        expected_svf = np.zeros((self.n_states, self.rollout_horizon))
        expected_svf[:, 0] = p_start_state

        if self.par:
            lock_expected_svf = Lock()

        ##Run MC rollouts parallely to calculate state visitation frequencies (run 100 threads parllely)
        rollouts = []
        for j in range(0, self.mc_rollouts, 100):
            if self.par:
                par_rollouts = 100 if self.mc_rollouts - j > 100 else self.mc_rollouts - j
                rollouts = []
                for i in range(par_rollouts):
                    rollout_i = Thread(name='rollout' + str(i), target=self.MC_rollout_par,
                                      args=(p_start_state, policy, expected_svf, lock_expected_svf))
                    rollouts.append(rollout_i)

                ##Start all threads
                for r in rollouts:
                    r.start()

                ##Wait for all threads to finish
                for r in rollouts:
                    r.join()
            else: ##Option not to parallel
                rollouts.append( self.MC_rollout(p_start_state, policy) )
               
        if not self.par: 
            for state_visitation_counts in rollouts:
                np.add(expected_svf, state_visitation_counts, expected_svf) ##Add in place, right?

        expected_svf[:, 1:] = expected_svf[:, 1:] / self.mc_rollouts
        return expected_svf.sum(axis=1), policy

    def find_start_state_freq(self):
        """
        Calculates the start state frequencies based on the trajectories.

        :return: The probability distribution over start states.
        :rtype: numpy.ndarray

        This method counts the number of occurrences of each start state in the list of trajectories, and returns the
        resulting probability distribution over start states. The start state frequency is calculated by dividing the
        count of each start state by the total number of trajectories.
        """

        start_state_count = np.zeros(self.n_states)
        for trajectory in self.trajectories:
            start_state_count[trajectory[0][0]] += 1
        p_start_state = start_state_count / len(self.trajectories)

        return p_start_state

    def MC_rollout(self, p_start_state, policy):
        """    
        Run a Monte Carlo (MC) rollout for a specified horizon and record the counts of visited states.

        :param p_start_state: A 1D NumPy array containing the probabilities of starting at each state.
        :type p_start_state: numpy.ndarray
        :param policy: A 2D NumPy array containing the policy probabilities for each state-action pair.
        :type policy: numpy.ndarray
        :return: A 2D NumPy array containing the count of times each state was visited at each time step.
        :rtype: numpy.ndarray

        This method runs a Monte Carlo (MC) rollout for a specified horizon and records the counts of visited states. The
        MC rollout starts from a randomly selected start state sampled according to the provided start state probabilities
        `p_start_state`, and then follows the provided policy `policy` for `rollout_horizon` time steps. At each time step,
        the MC rollout records the count of times each state was visited.
        """

        state_visitation_counts = np.zeros((self.n_states, self.rollout_horizon))

        ##Pick a start state
        state = np.random.choice(np.arange(self.n_states), p=p_start_state)

        for t in range(1, self.rollout_horizon):

            ##Reached terminal state 
            if state in self.terminal_states: break

            ##Select a valid action from policy
            action = np.random.choice(self.actions, p=policy[state][self.actions]/sum(policy[state][self.actions]))

            ##Execute action by selecting state from transition probabilities
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probability[state, action, :])

            ##Update state visitation count
            state_visitation_counts[next_state, t] += 1

            state = next_state

        return state_visitation_counts

    def MC_rollout_par(self, p_start_state, policy, expected_svf, lock_expected_svf):
        """
        Run a Monte Carlo (MC) rollout for a specified number of steps and record the counts of visited states in parallel.

        :param p_start_state: The probability distribution over start states.
        :type p_start_state: numpy.ndarray
        :param policy: The policy to execute the MC rollout.
        :type policy: numpy.ndarray
        :param expected_svf: The matrix to record the counts of visited states.
        :type expected_svf: numpy.ndarray
        :param lock_expected_svf: The lock object to ensure the thread-safe access to the expected_svf matrix.
        :type lock_expected_svf: threading.Lock

        :return: The state visitation counts.
        :rtype: numpy.ndarray

        This method runs a Monte Carlo (MC) rollout for a specified number of steps and records the counts of visited states
        in parallel using multiple threads. It first selects a start state from the probability distribution over start states.
        Then, at each time step, it selects a valid action from the policy and executes the action by selecting the next state from
        the transition probabilities. Finally, it updates the state visitation count matrix with the counts of visited states.
        """

        state_visitation_counts = np.zeros((self.n_states, self.rollout_horizon))

        ##Pick a start state
        state = np.random.choice(np.arange(self.n_states), p=p_start_state)

        for t in range(1, self.rollout_horizon):

            ##Reached terminal state 
            if state in self.terminal_states: break

            ##Select a valid action from policy
            action = np.random.choice(self.actions, p=policy[state][self.actions]/sum(policy[state][self.actions]))

            ##Execute action by selecting state from transition probabilities
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probability[state, action, :])

            ##Update state visitation count
            state_visitation_counts[next_state, t] += 1

            state = next_state

        ##Lock general expected_svf matrix
        lock_expected_svf.acquire()
        try:
            ##Add rollout counts to general matrix
            np.add(expected_svf, state_visitation_counts, expected_svf)
        finally:
            lock_expected_svf.release()
        return

    def find_expected_svf(self, r, p_start_state):
        """
        Direct calculation of state visitation frequencies (Algorithm 1 from the paper) without using MC rollouts.
        Returns the expected state visitation frequencies (SVF) vector and the policy vector.

        :param r: Reward vector for each state.
        :type r: numpy.ndarray
        :param p_start_state: Probability vector of each start state.
        :type p_start_state: numpy.ndarray

        :return: A tuple containing the expected state visitation frequencies vector and the policy vector.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)

        This method calculates the expected state visitation frequencies for the given reward vector and start state probabilities
        without using Monte Carlo rollouts. It returns a tuple containing the expected SVF vector and the policy vector.
        The method follows Algorithm 1 from the paper, where a stochastic policy is first calculated based on the reward vector
        and the transition probabilities. The expected SVF is then calculated iteratively by multiplying the previous iteration's
        expected SVF vector with the policy and transition probabilities at each time step. The final expected SVF is obtained by
        summing over all time steps.
        """

        ##Calculates probability for each action and state (lines 1-3 alg 1)
        policy = find_stochastic_policy(self.n_states, self.n_actions, self.transition_probability, r, self.terminal_states,
                             self.discount)

        expected_svf = np.tile(p_start_state, (self.rollout_horizon, 1)).T

        for t in range(1, self.rollout_horizon):
            expected_svf[:, t] = 0
            for i, j, k in product(range(self.n_states), range(self.n_actions), range(self.n_states)): #line 5 alg 1
                expected_svf[k, t] += (expected_svf[i, t-1] *
                                      policy[i, j] * ##Stochastic policy
                                       self.transition_probability[i, j, k])
                
        return expected_svf.sum(axis=1), policy


    def calculate_policy(self, n_actions, transition_probability, feature_matrix, trajectories, terminal_states):
        """
        Calculates the optimal policy based on learned rewards.

        :param n_actions: Number of actions
        :type n_actions: int
        :param transition_probability: Transition probabilities
        :type transition_probability: numpy.ndarray
        :param feature_matrix: Feature matrix
        :type feature_matrix: numpy.ndarray
        :param trajectories: Trajectories
        :type trajectories: list of numpy.ndarray
        :param terminal_states: Terminal states
        :type terminal_states: list of int
        :return: Optimal policy, rewards and state values
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """

        ##IRL rewards
        rewards, _, reward_weigths = self.irl(n_actions, transition_probability, feature_matrix, trajectories, terminal_states)
 
        ##Reconstruct policy based on learned rewards
        policy, v = find_policy(self.n_states, self.n_actions,self.transition_probability, rewards, self.terminal_states, self.discount)

        return policy, reward_weigths, v
