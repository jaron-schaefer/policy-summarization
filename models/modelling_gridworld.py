import sys
sys.path.append("..")
from gridworld import Gridworld
import numpy as np
from sklearn.preprocessing import StandardScaler


class ModelGridworld(Gridworld):

	def __init__(self, grid_list, reward_weights, trajectory_length = 5, discount = 1, rollout_horizon=20):
		"""Initialize the gridworld environment.

		:param grid_list: A list representing the gridworld.
		:type grid_list: list[str]
		:param reward_weights: The weights for different rewards.
		:type reward_weights: list[float]
		:param trajectory_length: The length of trajectories.
		:type trajectory_length: int
		:param discount: The discount factor for future rewards.
		:type discount: float
		:param rollout_horizon: The horizon for rollouts, defaults to 10.
		:type rollout_horizon: int, optional
		"""

		super().__init__(grid_list, reward_weights, discount)

		self.trajectory_length = trajectory_length
		self.discount = discount
		self.rollout_horizon = rollout_horizon

		self.features = self.get_features()

		self.standarized_features = self.standarized_state_features(self.features)

		# For machine Teaching
		self.feature_counts = self.get_state_action_fcounts()

		# All possible trajectories with state and action and only the states 
		self.candidate_trajectories, self.candidate_trajectories_indices = self.get_candidate_trajectories()



	def get_next_state_feature_vector( self , current_state , features):
		"""
		Get the feature vector for the next state given the current state.

		:param current_state: The index of the current state.
		:type current_state: int
		:return: The feature vector for the next state.
		:rtype: numpy.ndarray
		"""
		feature_vector = []
		for action in range( self.n_actions ):
			next_state = self.get_next_state( current_state , action )
			feature_vector += list( features[ next_state ])

		return np.array( feature_vector )


	def get_features(self):
		"""Get the features for every state in the gridworld.
		"""
		single_state_features = self.single_state_features
		#single_state_features = single_state_features.tolist()

		# get features of neighbors for each state
		neighbor_features = []
		for i in range( self.n_states):
			neighbor_features.append( self.get_next_state_feature_vector(i, single_state_features) )

		state_and_neighbor_features = np.array(np.concatenate( ( single_state_features , np.array(neighbor_features)) , axis=1 ))
		
		return np.array(state_and_neighbor_features)

	def run_mc_episode( self , state , action ):
		"""
			Runs Monte Carlo rollout for a given state-action pair.

			:param state: The starting state.
			:type state: int
			:param action: The action to be taken from the starting state.
			:type action: int
			:return: The Monte Carlo episode for the given state-action pair.
			:rtype: numpy.ndarray
		"""

		episode = [ self.features[ state ] ]

		for i in range( self.rollout_horizon - 1 ):
			
			state = self.get_next_state( state , action )
			action = self.policy[ state ]
			next_features = np.array(self.features[ state ])
			next_features_discounted = self.discount ** (i+1) * next_features
			episode.append(next_features_discounted)

		return np.array( episode )


	def get_state_action_fcounts( self ):
		"""
        Get feature counts for all unique states and actions in the grid.

        :return: A dictionary of feature counts for all unique state-action pairs.
        :rtype: dict
        """

		feature_counts = {}
		for state in range( self.n_states ):
			##For each action in state run mc rollout in simulator and calculate expected feature counts mu_sa (defined under eqation 4 in paper)
			for action in range( self.n_actions ):

				sa_features = self.run_mc_episode( state , action )
				mu_sa = np.sum( sa_features , axis=0 )

				##Add feature counts to dicationary
				feature_counts[ ( state , action ) ] = mu_sa

		return feature_counts

	def get_candidate_trajectories( self ):
		"""
		Generate candidate trajectories of length `trajectory_length` for all states in the grid following the current policy.

		:return: A list of candidate trajectories, each of which is a tuple of (List of state features, List of actions,
		List of identifiers , List of rewards) for each trajectory.
		:rtype: list
		"""

		candidate_trajectories = []  
		candidate_trajectories_indices = []  

		##Generate a trajectory of maximal length trajectory_length from each state in the grid following policy
		for i in range(self.n_states):
			current_state = i

			trajectory = []
			trajectory_indices = []
			nr_steps = 0
			# Play out policy for trajectory_length steps
			for j in range(self.trajectory_length):  

				if current_state not in self.terminal_indices:
					nr_steps += 1

				action = self.policy[current_state] 

				trajectory.append((current_state, action))
				trajectory_indices.append(current_state)
				current_state = self.get_next_state(current_state, action)

			if nr_steps < 2:
				continue
			candidate_trajectories.append(trajectory)
			candidate_trajectories_indices.append(trajectory_indices)
		

		return np.array(candidate_trajectories), np.array(candidate_trajectories_indices)


	def standarized_state_features(self, X , unlabeled_indices=[] , test_indices=[] ):
		"""Standardizes the features in X using the Standard Scaler method from `sklearn.preprocessing`.

		:param X: The input feature matrix to be standardized.
		:type X: numpy.ndarray
		:param unlabeled_indices: A list of indices corresponding to unlabeled examples in X. Default is an empty list.
		:type unlabeled_indices: list, optional
		:param test_indices: A list of indices corresponding to test examples in X. Default is an empty list.
		:type test_indices: list, optional
		:return: The standardized feature matrix.
		:rtype: numpy.ndarray
		"""

		standardized_X = np.copy( X )

		test_indices = list( unlabeled_indices ) + list( test_indices )
		train_indices = list( np.setdiff1d( np.arange( X.shape[ 0 ] ) , test_indices ) )
		scaler = StandardScaler()
		standardized_X[ train_indices ] = scaler.fit_transform( X[ train_indices , : ] )

		if len( test_indices ):
			standardized_X[ test_indices ] = scaler.transform( X[ test_indices , : ] )

		return standardized_X
