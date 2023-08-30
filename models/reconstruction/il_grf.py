import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel


class GRF:

	def __init__( self , kernel_params , n_classes ):
		"""
        Initialize the GRF classifier with the given parameters.

        If the kernel type is "rbf", "poly", or "linear", the corresponding kernel function is used.

        If `fit_kernel` is True, it fits Gaussian process classifiers for each binary classification problem.
        `X` and `y` must be passed in.

        :param kernel_params: A dictionary containing the kernel function parameters.
        :type kernel_params: dict
        :param n_classes: The number of classes or actions.
        :type n_classes: int
        """

		if kernel_params[ "kernel" ] == "rbf":
			kernel = rbf_kernel
		elif kernel_params[ "kernel" ] == "poly":
			kernel = polynomial_kernel
		elif kernel_params[ "kernel" ] == "linear":
			kernel = linear_kernel
		else:
			assert False , "Invalid kernel in grf: " 

		self.kernel_params = {}
		for key , val in kernel_params.items():
			if key != "kernel":
				self.kernel_params[ key ] = val

		self.n_classes = n_classes
		
		self.kernels = []
		for k in range( self.n_classes ):
			self.kernels.append( kernel )

	def predict_proba_binary( self , laplacian , labeled , unlabeled , labels ):
		"""
		Predict the probabilities of binary labels for unlabeled data given the labeled data and Laplacian matrix.

		:param laplacian: The Laplacian matrix of the graph constructed from the labeled and unlabeled data.
		:type laplacian: numpy.ndarray
		:param labeled: The indices of the labeled data in the feature matrix.
		:type labeled: numpy.ndarray
		:param unlabeled: The indices of the unlabeled data in the feature matrix.
		:type unlabeled: numpy.ndarray
		:param labels: The binary labels of the labeled data.
		:type labels: numpy.ndarray
		:return: The predicted probabilities of the binary labels for the unlabeled data.
		:rtype: numpy.ndarray
		"""

		self.l_uu = laplacian[ unlabeled , : ][ : , unlabeled ]
		
		try:
			self.l_uu_inv = np.linalg.inv( self.l_uu )
		except:
			raise ValueError("Laplacian matrix was not invertable")

		self.l_ul = laplacian[ unlabeled , : ][ : , labeled ]
		self.unlabeled_prev = unlabeled
		f_u = np.matmul( np.matmul( ( -1 * self.l_uu_inv ) , self.l_ul ) , labels )
		return f_u

	def predict_probas( self , X , y , unlabeled ):
		"""
        Predict the class probabilities for the unlabeled data using the fitted Gaussian Random Fields model.
        
        :param X: The feature matrix of shape (n_states, n_features).
        :type X: numpy.ndarray
        :param y: The label matrix of shape (n_states, n_classes).
        :type y: numpy.ndarray
        :param unlabeled: The indices of the unlabeled data points.
        :type unlabeled: numpy.ndarray
        
        :return: The class probabilities for the unlabeled data of shape (n_unlabeled, n_classes).
        :rtype: numpy.ndarray
        """

		self.X = X
		self.y = y

		self.binary_class_labels = []
		for k in range( self.n_classes ):
			binary_labels = np.zeros( ( y.shape[ 0 ] , 2 ) )
			binary_labels[ np.where( y == k) , 1 ] = 1
			binary_labels[ np.where( y != k) , 0 ] = 1
			self.binary_class_labels.append( binary_labels )
		self.binary_class_labels = np.array( self.binary_class_labels )
		labeled = np.setdiff1d( np.arange( self.X.shape[ 0 ] ) , unlabeled )
		
		all_class_probas = []
		for i in range( self.n_classes ):
			binary_labels = self.binary_class_labels[ i ][ labeled , : ]
			laplacian = self.compute_laplacian( self.kernels[ i ] , X )
			class_probas = self.predict_proba_binary( laplacian , labeled , unlabeled , binary_labels )[ : , 1 ]

			all_class_probas.append( class_probas )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )
		self.f_u = all_class_probas
		self.unlabeled = unlabeled

		return all_class_probas

	def predict( self , X , y , unlabeled ):
		"""
		Predict the labels (actions) of the unlabeled data (states) using the fitted Gaussian Random Fields model.

        :param X: The feature matrix of shape (n_states, n_features).
        :type X: numpy.ndarray
        :param y: The label matrix of shape (optimal actions/policy)
        :type y: numpy.ndarray
        :param unlabeled: The indices of the unlabeled data points.
        :type unlabeled: numpy.ndarray
        """

		all_class_probas = self.predict_probas( X , y , unlabeled )

		predictions = []
		for class_probas in all_class_probas:
			predictions.append( np.random.choice( np.where( class_probas == np.max( class_probas ) )[ 0 ] ) )

		return np.array( predictions )

	def predict_proba_binary_update( self , query_idx , f_u , labels ):
		"""
		Compute the updated predicted probability of binary classification using the queried data point.
		
		:param query_idx: The index of the queried data point in the unlabeled data.
		:type query_idx: int
		:param f_u: The predicted probability of binary classification for the unlabeled data points.
		:type f_u: numpy.ndarray
		:param labels: The binary class labels for the labeled data points.
		:type labels: numpy.ndarray
		:return: The updated predicted probability of binary classification for the unlabeled data points and the index of the queried data point in the unlabeled data.
		:rtype: tuple(numpy.ndarray, int)
		"""

		unlabeled_query_idx = list( self.unlabeled ).index( query_idx ) # Get index of query id in unlabeled array
		try:
			f_u = f_u + ( np.argmax( labels[ query_idx ] ) - f_u[ unlabeled_query_idx ] ) * ( self.l_uu_inv[ : , unlabeled_query_idx ] / self.l_uu_inv[ unlabeled_query_idx , unlabeled_query_idx ] )
		except Exception:
			raise RuntimeError("Matrix multiplication failed")
		return f_u , unlabeled_query_idx

	def predict_with_new_label( self , query_idx ):
		"""
		Predict the labels (actions) of the unlabeled data (states) after assigning a new label to a single query state.

		:param query_idx: The index of the query state to which a new label has been assigned.
		:type query_idx: int
		:return: The predicted labels (actions) of the unlabeled data (states) after assigning a new label to the query state.
		:rtype: numpy.ndarray
		"""
		assert self.f_u is not None , 'Must predict before calling predict_with_new_label'

		all_class_probas = []
		for i in range( len( self.binary_class_labels ) ):
			f_u , unlabeled_query_idx = self.predict_proba_binary_update( query_idx , self.f_u[ : , i ] , self.binary_class_labels[ i ] )
			f_u = np.delete( f_u , unlabeled_query_idx )
			all_class_probas.append( f_u )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )

		predictions = np.argmax( all_class_probas , axis=1 )

		return predictions

	def predict_with_new_trajectory( self , query_indices ):
		"""
		Predict the labels of a new trajectory by updating the Gaussian Random Fields model with the newly labeled data points.

		:param query_indices: The indices of the newly labeled data points.
		:type query_indices: List[int]
		:return: The predicted labels for the new trajectory.
		:rtype: numpy.ndarray
		"""

		assert self.f_u is not None , 'Must predict before calling predict_with_new_label'

		all_class_probas = []
		for i in range( len( self.binary_class_labels ) ):
			f_u = np.copy( self.f_u[ : , i ] )
			unlabeled_query_indices = []
			for query_idx in query_indices:
				f_u , unlabeled_query_idx = self.predict_proba_binary_update( query_idx , f_u , self.binary_class_labels[ i ] )
				unlabeled_query_indices.append( unlabeled_query_idx )
			f_u = np.delete( f_u , unlabeled_query_indices )
			all_class_probas.append( f_u )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )

		predictions = np.argmax( all_class_probas , axis=1 )

		return predictions

###################################################################HELPERS##################################################################################

	def compute_laplacian( self , kernel , X ):
		"""
		Compute the combinatorial Laplacian matrix of a graph defined by a kernel matrix.
		
		:param kernel: A function that computes a kernel matrix given a matrix of data points X.
		:type kernel: callable
		:param X: An array-like object of shape (n_samples, n_features) that represents the data points.
		:type X: array-like
		:return: The combinatorial Laplacian matrix of the graph.
		:rtype: numpy.ndarray
		"""
		# Compute the pairwise kernel matrix for the data.
		G = kernel( X , **self.kernel_params )

		# Compute the weighted degree matrix, which is a diagonal matrix with
		# the sum of the weights of each node on its diagonal.
		D = np.diag( np.ones( G.shape[ 0 ] ) )
		for i in range( D.shape[ 0 ] ):
			D[ i , i ] = np.sum( G[ i , : ] )

		# Compute the combinatorial Laplacian matrix by subtracting the weighted
		# adjacency matrix from the weighted degree matrix.
		combinatorial_laplacian = D - G

		return combinatorial_laplacian
###################################################################HELPERS##################################################################################

