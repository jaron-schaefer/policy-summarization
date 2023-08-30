"""
Implements the active learning algorithm by Zhu, Lafferty, et al. (2003) (refer to  Lage et al. (2019))
"""

import numpy as np
from sklearn.metrics import accuracy_score

class ActiveLearning:

	def __init__( self , model , n_queries=None , min_accuracy=None , strategy="active" ):
		"""
        Initializes the ActiveLearning object.

        :param model: The model used for active learning.
        :type model: object
        :param n_queries: The number of queries to make during active learning. Either `n_queries` or `min_accuracy` should be provided.
        :type n_queries: int, optional
        :param min_accuracy: The minimum accuracy threshold to achieve during active learning. Either `n_queries` or `min_accuracy` should be provided.
        :type min_accuracy: float, optional
        :param strategy: The active learning strategy to use. Default is "active".
        :type strategy: str, optional
        :raises AssertionError: If neither `n_queries` nor `min_accuracy` is provided, or if both are provided.
        """

		assert n_queries or min_accuracy , "Must pass in a number of steps (n_steps=) or an accuracy threshold! (min_accuracy=)"
		assert n_queries is None or min_accuracy is None , "Must only pass in one of number of steps (n_steps=) and accuracy threshold! (min_accuracy=)"

		self.n_queries = n_queries
		self.min_accuracy = min_accuracy
		self.model = model
		self.strategy = strategy

	def fit( self , X , y , trajectory_indices=[] ):
		"""
		Fits the active learning model to the provided data.

		:param X: The input data.
		:type X: array-like
		:param y: The target labels corresponding to the input data.
		:type y: array-like
		:param trajectory_indices: Optional list of trajectory indices to optimize over. If not provided, the optimization is performed directly over X. Default is an empty list.
		:type trajectory_indices: list, optional
		"""
		"""
		Assumptions: X has each state once
					 y has the corresponding (single) action for each state
					 trajectory_indices has a list of trajectories in the form 
					 of indices into X.  This is what the optimization is over.
					 This is optional, if it's not included, we search over X directly
		"""

		if len( trajectory_indices ) == 0: ##If there are no trajectory indices, index into X directly
			trajectory_indices = np.array([ [ i ] for i in range( X.shape[ 0 ] ) ])

		##Holds onto trajectories used in learning and ones left to query
		queried_indices_into_trajectory_indices = []
		remaining_indices_into_trajectory_indices = list( np.arange( len( trajectory_indices ) ) )

		##Indices in X that haven't been queried yet -- others are into trajectory indices!!
		heldout_indices_into_X = list( np.arange( X.shape[ 0 ] ) )

		##Accuracy of all and accuracy of remaining!
		heldout_accuracies = []
		accuracies = []

		##While you haven't taken enough steps or you haven't reached the right accuracy
		while ( ( self.n_queries is not None and len( accuracies ) < self.n_queries ) or \
				( self.min_accuracy is not None and ( len( accuracies ) == 0 or accuracies[ -1 ] < self.min_accuracy ) ) ):
			##Query next point from amongst reminaing points and updated queried and remaining trajectories
			query_idx = self.determine_next_query( X , y , remaining_indices_into_trajectory_indices , trajectory_indices , heldout_indices_into_X )
			
			remaining_indices_into_trajectory_indices = [ i for i in remaining_indices_into_trajectory_indices if i != query_idx ]
			queried_indices_into_trajectory_indices += [ query_idx ]
			##Get all ids not in any of the queried trajectories
			heldout_indices_into_X = [ i for i in range( X.shape[ 0 ] ) if i not in np.ravel( trajectory_indices[ queried_indices_into_trajectory_indices ] ) ]
			##Predict on heldout points!
			heldout_predictions = self.model.predict( X , y , heldout_indices_into_X )

			##For points we've seen, use label, else use predictions
			all_predictions = np.copy( y )
			all_predictions[ heldout_indices_into_X ] = heldout_predictions

			##Compute accuracies
			heldout_accuracies.append( accuracy_score( heldout_predictions , y[ heldout_indices_into_X ] ) )
			accuracies.append( accuracy_score( all_predictions , y ) ) ##All points

		##Save stuff
		self.queried_trajectory_indices = queried_indices_into_trajectory_indices
		self.remaining_trajectory_indices = remaining_indices_into_trajectory_indices
		self.heldout_accuracies = heldout_accuracies
		self.accuracies = accuracies

	def determine_next_query( self , X , y , remaining_indices_into_trajectory_indices , trajectory_indices , heldout_indices_into_X ):
		"""
		Determines the next query point for active learning.

		:param X: The input data.
		:type X: array-like
		:param y: The target labels corresponding to the input data.
		:type y: array-like
		:param remaining_indices_into_trajectory_indices: Indices into trajectory_indices for the remaining trajectories to consider.
		:type remaining_indices_into_trajectory_indices: list
		:param trajectory_indices: Indices representing the trajectories.
		:type trajectory_indices: array-like
		:param heldout_indices_into_X: Indices into X that have not been queried yet.
		:type heldout_indices_into_X: list
		:return: The index of the next query point.
		:rtype: int
		"""
		##If we have a random acquisition strategy, return a random trajectory -- Could make this smarter!!
		if self.strategy == "random":
			return np.random.choice( remaining_indices_into_trajectory_indices )

		##Find points in X that have been queried!
		queried_indices_into_X = [ i for i in np.unique( np.ravel( trajectory_indices ) ) if i not in heldout_indices_into_X ] ##Indices into X not in heldout

		##Initialize predictions -- predict step in inner loop assumes they're initialized and adds a small set of points to update
		predictions = self.model.predict( X , y , heldout_indices_into_X )

		risks = [] ##For deciding on the query -- 1 - accuracy with new trajectory
		for indices in trajectory_indices[ remaining_indices_into_trajectory_indices ]:

			##Find only the as of yet unlabeled indices in the trajectory!
			newly_queried_indices = [ i for i in indices if i not in queried_indices_into_X ]

			##Local copies of the queries and heldouts we're trying in this iteration
			modified_queried_indices_into_X = [ i for i in queried_indices_into_X ] + newly_queried_indices ##Copy locally + newly queried points
			modified_heldout_indices_into_X = [ i for i in heldout_indices_into_X if i not in indices ] ##Copy locally - newly queried points

			if len( newly_queried_indices ) > 0: ##Only bother if there are new points in the trajectory
				##Make predictions with new points!
				predictions = self.model.predict_with_new_trajectory( newly_queried_indices )

				##Compute risk as 1 - accuracy of remaining unlabeled!
				risks.append( 1 - accuracy_score( predictions , y[ modified_heldout_indices_into_X ] ) )
			else: ##Don't add trajetcories with no new points!!
				risks.append( np.inf )

		##Shuffle indices to introduce randomness
		indices = np.arange( len( risks ) )
		np.random.shuffle( indices )
		risks = np.array( risks )[ indices ]

		return remaining_indices_into_trajectory_indices[ indices[ np.argmin( risks ) ] ]


