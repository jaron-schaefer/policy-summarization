import numpy as np
from models.value_iteration import *

class Gridworld:

	def __init__( self, grid_list, reward_weights, discount = 1):

		if not all(len(row) == len(grid_list[0]) for row in grid_list):
			raise ValueError("All rows in the grid must have the same length")

		self.action_names = ['Up' , 'Down' , 'Left' , 'Right' ]
		self.n_actions = 4
		self.grid_list = grid_list.copy()
		self.grid_list.reverse()
		self.discount = discount

		##Grid size
		self.max_x = len(grid_list[0].split())
		self.max_y = len(grid_list)
		self.grid_size = (self.max_x,self.max_y)
		self.n_states = self.max_x * self.max_y ##Size of the grid

		# make state_identifiers
		self.state_identifiers = []
		for x in range( self.max_x ):
			for y in range( self.max_y ):
				self.state_identifiers.append( [ x , y ] )
			
		self.state_indices = np.arange( len( self.state_identifiers ) )
		
		self.terminal_indices = []
        
		self.single_state_features = self.get_single_state_features()

		assert len(self.terminal_indices) != 0, "Gridworld needs at least one terminal state"
		assert len(reward_weights) == 9, "Wrong reward weights length"
		# set reward for each typ of cell
		self.reward_weights = reward_weights 
	
		# get reward for every state
		self.rewards = np.dot(self.single_state_features, self.reward_weights)

		# Transition probabilities for all states - deterministic for now
		self.transition_probas = self.get_transition_probabilities()

		##Get optimal policy for selected reward weights and features 
		self.policy, self.values = self.get_policy()



	def get_next_state( self , current_state , action ):
		"""
		Calculate the next state based on the current state and the action taken.

		:param current_state: An integer representing the current state.
		:type current_state: int

		:param action: An integer representing the action taken.
		:type action: int

		:return: An integer representing the next state after taking the given action.
		:rtype: int
		"""

		current_state_identifiers = self.state_identifiers[ current_state ] 
		next_state_identifiers = [ current_state_identifiers[ 0 ] , current_state_identifiers[ 1 ] ] ##Initialize to current state_identifiers
		
		if current_state in self.terminal_indices:
			return current_state
		# Up
		if action == 0:
			new_y = current_state_identifiers[ 1 ] + 1
			if new_y < self.max_y:
				next_state_identifiers[ 1 ] = new_y
		# Down
		elif action == 1:
			new_y = current_state_identifiers[ 1 ] - 1
			if new_y >= 0:
				next_state_identifiers[ 1 ] = new_y
		# Left
		elif action == 2:
			new_x = current_state_identifiers[ 0 ] - 1
			if new_x >= 0:
				next_state_identifiers[ 0 ] = new_x
		# Right
		elif action == 3:
			new_x = current_state_identifiers[ 0 ] + 1
			if new_x < self.max_x:
				next_state_identifiers[ 0 ] = new_x
		else:
			assert False , "Grid world only has 4 actions!"

		next_state = self.state_identifiers.index( next_state_identifiers )

		return next_state


	def get_transition(self, state_i, state_k, action):
		"""
		Get the probability of moving from state i to state k under action a (deterministic).

		:param state_i: An integer representing the current state.
		:type state_i: int

		:param state_k: An integer representing the next state.
		:type state_k: int

		:param action: An integer representing the action taken.
		:type action: int

		:return: A float representing the probability of moving from state i to state k under action a.
		:rtype: float
		"""

		next_state = self.get_next_state(state_i, action)
		##state_k is the intended state to move to
		if next_state == state_k:
			return 1.0
		##illegal action in state (takes out of grid) - stay in origin state
		elif next_state == state_i and state_i == state_k:
			return 1.0
		else:
			return 0.0


	def get_transition_probabilities(self):
		"""
		Get the transition probabilities for all states.

		:return: A 3D numpy array containing the transition probabilities.
		:rtype: numpy.ndarray
   		 """

		transition_probability = np.array(
			[[[self.get_transition(state_i, state_k, action_a)
			   for state_k in range(self.n_states)]
			  for action_a in range(self.n_actions)]
			 for state_i in range(self.n_states)])

		return transition_probability

	def get_single_state_features(self):
		"""
		Get the single-state features of the gridworld.

		:raises ValueError: If there is an invalid grid element in the gridworld.
		:return: A numpy array containing the single-state features.
		:rtype: numpy.ndarray
		"""
		
		single_state_features = []

		for x in range( self.max_x ):
			for y in range( self.max_y ):
				row_str = self.grid_list[y].split()
				cell_str = row_str[x]
				if cell_str == "-":
					single_state_features.append([1,0,0,0,0,0,0,0,0])
				elif cell_str == "A":
					single_state_features.append([0,1,0,0,0,0,0,0,0])
					self.terminal_indices.append(self.state_identifiers.index([x,y]))
				elif cell_str == "B":
					single_state_features.append([0,0,1,0,0,0,0,0,0])
				elif cell_str == "C":
					single_state_features.append([0,0,0,1,0,0,0,0,0])
				elif cell_str == "D":
					single_state_features.append([0,0,0,0,1,0,0,0,0])
				elif cell_str == "E":
					single_state_features.append([0,0,0,0,0,1,0,0,0]) 
				elif cell_str == "F":
					single_state_features.append([0,0,0,0,0,0,1,0,0])
				elif cell_str == "G":
					single_state_features.append([0,0,0,0,0,0,0,1,0])
				elif cell_str == "H":
					single_state_features.append([0,0,0,0,0,0,0,0,1])
				else:
					raise ValueError(f"Invalid grid element '{cell_str}' at row {y+1}, column {x+1}")
		return np.array(single_state_features)
	
	def get_policy(self):
		"""
		Calculate the optimal policy for the gridworld using value iteration.

		:return: A tuple containing the optimal policy, the corresponding value function for each state, and a list of states with multiple indices.
		:rtype: tuple(numpy.ndarray, numpy.ndarray, list[int])
		"""

		policy, values = find_policy(self.n_states, self.n_actions, self.transition_probas, self.rewards, self.terminal_indices,
													 discount=self.discount)
		return policy, values
	
	def get_optimal_path(self, state):
		optimal_path = [] 
		optimal_reward = 0
		for i in range(100):
			if state in self.terminal_indices:
				break
			action = self.policy[state]
			optimal_path.append([state, int(action)])
			state = self.get_next_state(state, action)
			optimal_reward += self.rewards[state]
		return optimal_path, int(optimal_reward)
	
	def get_path_to_goal(self, policy, start_state):
		"""
		Returns the path and total reward obtained by following a given policy from a start state.
		max length without reaching the goal state = 50 

		:param policy: The policy mapping states to actions.
		:type policy: dict
		:param start_state: The starting state.
		:type start_state: int
		:return: The path as a list of state-action pairs and the total reward obtained.
		:rtype: tuple
		"""
		
		path = []
		total_reward = 0
		state = start_state
		for i in range(50):
			if state in self.terminal_indices:
				break
			action = policy[state]
			path.append([state, int(action)])
			state = self.get_next_state(state, action)
			total_reward += self.rewards[state]

		return path, int(total_reward)

	
