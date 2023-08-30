import os
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from models.reconstruction.il_grf import GRF
from models.reconstruction.irl_max_ent import Maxent
from models.extraction.il_active_learning import ActiveLearning
from models.extraction.irl_machine_teaching import MachineTeaching
from models.modelling_gridworld import ModelGridworld


class HyperparameterComparisonPipeline:
	"""
		Calculates the reconstruction accuracy for each combination of the given parameters for the gridworld and saves it into csv files
	"""
	
	def __init__( self, gridworld_hyperparams=[] , il_extraction_hyperparams= [], irl_extraction_hyperparams = [], il_reconstruction_hyperparams = [], irl_reconstruction_hyperparams = []):

		self.gridworld_hyperparams = [ params for params in ParameterGrid( gridworld_hyperparams ) ]

		self.il_extraction_hyperparams_list = [ params for params in ParameterGrid( il_extraction_hyperparams ) ]
		self.irl_extraction_hyperparams_list = [ params for params in ParameterGrid( irl_extraction_hyperparams ) ]

		self.il_reconstruction_hyperparams_list = [ params for params in ParameterGrid(il_reconstruction_hyperparams ) ]
		self.irl_reconstruction_hyperparams_list = [ params for params in ParameterGrid(irl_reconstruction_hyperparams ) ]

		self.summary_length = self.il_extraction_hyperparams_list[0]["summary_length"]
	
	def compute_reconstruction_accuracies(self, variation_name:str):

		data_path =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'model_data', 'test')
		
		if not os.path.exists(data_path):
			os.makedirs(data_path)
		
		file_path_il_il = os.path.join(data_path, variation_name +'_IL_IL.csv')
		file_path_il_irl = os.path.join(data_path, variation_name +'_IL_IRL.csv')
		file_path_irl_irl = os.path.join(data_path, variation_name +'_IRL_IRL.csv')
		file_path_irl_il = os.path.join(data_path, variation_name +'_IRL_IL.csv')
		assert all(not os.path.isfile(file_path) for file_path in [file_path_il_il, file_path_il_irl, file_path_irl_irl, file_path_irl_il]), "One or more output files already exist"


		self.il_il_results = []
		self.il_irl_results = []
		self.irl_irl_results = []
		self.irl_il_results = []

		# Use IL sumamry
		for gridworld_params in self.gridworld_hyperparams:
			self.gridworld = ModelGridworld(**gridworld_params)

			for extract_hyperparams in tqdm(self.il_extraction_hyperparams_list):
				try:
					summary_trajectories, labeled_states , unlabeled_states = self._extract_summary( extract_hyperparams )	
				except Exception as e:
					print(variation_name + " threw " + type(e).__name__)
					continue
				
				# get IL reconstruction
				for reconstruct_hyperparams in self.il_reconstruction_hyperparams_list:
					try:
						reconstructed_policy, predictions_of_actions_for_unlabeled, inferred_reward_weights = self._compute_reconstruction( reconstruct_hyperparams , summary_trajectories,  unlabeled_states)
						accuracy = accuracy_score( self.gridworld.policy[ unlabeled_states] , predictions_of_actions_for_unlabeled)
						self.il_il_results.append([gridworld_params, extract_hyperparams, reconstruct_hyperparams,reconstructed_policy, summary_trajectories.tolist(), inferred_reward_weights, accuracy])
					except Exception as e:
						print(variation_name + " threw " + type(e).__name__)
						continue

				# get IRL reconstruction
				for reconstruct_hyperparams in self.irl_reconstruction_hyperparams_list:
					try:
						reconstructed_policy, predictions_of_actions_for_unlabeled, inferred_reward_weights = self._compute_reconstruction( reconstruct_hyperparams , summary_trajectories,  unlabeled_states)
						accuracy = accuracy_score( self.gridworld.policy[ unlabeled_states] , predictions_of_actions_for_unlabeled)
						self.il_irl_results.append([gridworld_params, extract_hyperparams, reconstruct_hyperparams, reconstructed_policy, summary_trajectories.tolist(), inferred_reward_weights, accuracy])
					except Exception as e:
						print(variation_name + " threw " + type(e).__name__)
						continue

		il_il_recon_df = pd.DataFrame(self.il_il_results, columns=['gridworld_params', 'extraction_hyperparams', 'reconstruction_hyperparams','reconstructed_policy', 'summary', 'inferred_reward_weights', 'accuracy'])
		il_irl_recon_df = pd.DataFrame(self.il_irl_results, columns=['gridworld_params', 'extraction_hyperparams', 'reconstruction_hyperparams','reconstructed_policy', 'summary','inferred_reward_weights',  'accuracy'])
		il_il_recon_df.to_csv(file_path_il_il, index=False)
		il_irl_recon_df.to_csv(file_path_il_irl, index=False)

		# Use IRL sumamry
		for gridworld_params in self.gridworld_hyperparams:
			self.gridworld = ModelGridworld(**gridworld_params)

			for extract_hyperparams in tqdm(self.irl_extraction_hyperparams_list):
				
				try:
					summary_trajectories, labeled_states , unlabeled_states = self._extract_summary( extract_hyperparams )	
				except Exception as e:
					print(variation_name + " threw " + type(e).__name__)
					continue

				# get IL reconstruction
				for reconstruct_hyperparams in self.il_reconstruction_hyperparams_list:
					try:
						reconstructed_policy, predictions_of_actions_for_unlabeled, inferred_reward_weights = self._compute_reconstruction( reconstruct_hyperparams , summary_trajectories,  unlabeled_states)
						accuracy = accuracy_score( self.gridworld.policy[ unlabeled_states] , predictions_of_actions_for_unlabeled)
						self.irl_il_results.append([gridworld_params, extract_hyperparams, reconstruct_hyperparams,reconstructed_policy, summary_trajectories.tolist(), inferred_reward_weights, accuracy])
					except Exception as e:
						print(variation_name + " threw " + type(e).__name__)
						continue

				# get IRL reconstruction
				for reconstruct_hyperparams in self.irl_reconstruction_hyperparams_list:
					try:
						reconstructed_policy, predictions_of_actions_for_unlabeled, inferred_reward_weights = self._compute_reconstruction( reconstruct_hyperparams, summary_trajectories,  unlabeled_states)
						accuracy = accuracy_score( self.gridworld.policy[ unlabeled_states] , predictions_of_actions_for_unlabeled)
						self.irl_irl_results.append([gridworld_params, extract_hyperparams, reconstruct_hyperparams,reconstructed_policy, summary_trajectories.tolist(), inferred_reward_weights, accuracy])
					except Exception as e:
						print(variation_name + " threw " + type(e).__name__)
						continue

		irl_irl_recon_df = pd.DataFrame(self.irl_irl_results, columns=['gridworld_params', 'extraction_hyperparams', 'reconstruction_hyperparams','reconstructed_policy', 'summary','inferred_reward_weights',  'accuracy'])
		irl_il_recon_df = pd.DataFrame(self.irl_il_results, columns=['gridworld_params', 'extraction_hyperparams', 'reconstruction_hyperparams','reconstructed_policy', 'summary','inferred_reward_weights',  'accuracy'])
		irl_irl_recon_df.to_csv(file_path_irl_irl, index=False)
		irl_il_recon_df.to_csv(file_path_irl_il, index=False)


	def _instantiate_model_from_hyperparams( self , hyperparams ):
		"""
		Instantiates a model object from hyperparameters specified in a dictionary.
		
		:param hyperparams: A dictionary containing hyperparameters for the model instantiation.
		:type hyperparams: dict
		
		:return: An object of the specified model class instantiated with the given hyperparameters.
		:rtype: object
		"""
		if hyperparams[ "model" ] == "il":
			reconstruction_model = GRF( hyperparams[ "kernel_params" ] , n_classes=self.gridworld.n_actions )
			
			if hyperparams[ "algorithm" ] == "extraction":
				if hyperparams[ "strategy" ] == "al":
					model = ActiveLearning( model=reconstruction_model  , n_queries=self.summary_length, strategy="al")
				elif hyperparams[ "strategy" ] == "random":
					model = ActiveLearning( model=reconstruction_model  , n_queries=self.summary_length, strategy="random" ) ##Random extraction!!
				else:
					assert False , str( hyperparams )
			else:
				model = reconstruction_model

		elif hyperparams[ "model" ] == "irl":
			
			if hyperparams[ "algorithm" ] == "reconstruction":  # IRL Reconstruction
				hyperparams[ "maxent_params" ]["discount"] = self.gridworld.discount
				model = Maxent( **hyperparams[ "maxent_params" ])
			else: # During MT extraction
				model = None
		else:
			assert False , str( hyperparams[ "model" ] )
		
		return model


	def _extract_summary( self , extract_hyperparams ):
		"""
		Extracts a summary of the candidate trajectories using the given hyperparameters for the extraction model.

		:param extract_hyperparams: A dictionary containing hyperparameters for the extraction model.
		:type extract_hyperparams: dict

		:returns: A tuple containing the summary trajectories, the states included in the summary, and the states not included in the summary.
		:rtype: tuple
		:raises AssertionError: If `extract_hyperparams["algorithm"]` is not "extraction".

		:raises AssertionError: If `extract_hyperparams["model"]` is not "il" or "irl".

		:raises AssertionError: If `extract_hyperparams["model"]` is not implemented.

		"""
		assert extract_hyperparams["algorithm"] == "extraction", "Not extraction params"

		extraction_model = self._instantiate_model_from_hyperparams( extract_hyperparams )

		if extract_hyperparams[ "model" ] == "il":
	
			extraction_model.fit( X = self.gridworld.standarized_features, y = self.gridworld.policy, trajectory_indices = self.gridworld.candidate_trajectories_indices)
			trajectory_indices_of_summary = extraction_model.queried_trajectory_indices # chosen trajectories as indices
			summary_trajectories = self.gridworld.candidate_trajectories[trajectory_indices_of_summary] # trajectories of the summary

			# all states that are visited in the summary
			in_summary_visited_states = []
			for traj in summary_trajectories:
				for state, action in traj:
					in_summary_visited_states.append(state)

			labeled_states = np.unique( in_summary_visited_states ) # states that are included in the summary 
			unlabeled_states = np.setdiff1d( np.arange( self.gridworld.standarized_features.shape[ 0 ] ) , labeled_states) # states in X not in the summary		
		elif extract_hyperparams[ "model" ] == "irl":
			machine_teaching = MachineTeaching( init_states = self.gridworld.state_indices , optimal_actions = self.gridworld.policy, sa_fcounts = self.gridworld.feature_counts, 
												num_actions = self.gridworld.n_actions, candidate_trajectories = self.gridworld.candidate_trajectories, summary_length = self.summary_length)
			
			summary_trajectories = np.array(machine_teaching.set_cover_optimal_teaching()) # trajectories of the summary
			trajectory_indices_of_summary = [array[0][0] for array in summary_trajectories] # chosen trajectories as indices

			# all states that are visited in the summary
			in_summary_visited_states = []
			for traj in summary_trajectories:
				for state, action in traj:
					in_summary_visited_states.append(state)

			labeled_states = np.unique( in_summary_visited_states ) # states that are included in the summary 
			unlabeled_states = np.setdiff1d( np.arange( self.gridworld.standarized_features.shape[ 0 ] ) , labeled_states) # states in X not in the summary		

		else:
			assert False , "Only grf, irl, extraction implemented, not " + str( extract_hyperparams[ "model" ] )
		
		return summary_trajectories , labeled_states , unlabeled_states


	def _compute_reconstruction( self , reconstruct_hyperparams , reconstruction_trajectories,  unlabeled_indices ):
		"""
		Computes the reconstruction of a policy based on the given reconstruction hyperparameters and trajectories.

		:param reconstruct_hyperparams: A dictionary containing the hyperparameters for the reconstruction algorithm.
		:type reconstruct_hyperparams: dict

		:param reconstruction_trajectories: A list of trajectories to use for the reconstruction. Those are the summary trajectories
		:type reconstruction_trajectories: list

		:param unlabeled_indices_into_unique_states: A list of indices into unique states that are unlabeled.
		:type unlabeled_indices_into_unique_states: list

		:return: The reconstructed policy. Predictions for unlabeld states, inferred_reward_weights
		:rtype: numpy.ndarray
		"""

		assert reconstruct_hyperparams["algorithm"] == "reconstruction", "Not reconstruction params"
		reconstruction_model = self._instantiate_model_from_hyperparams( reconstruct_hyperparams )

		if reconstruct_hyperparams[ "model" ] == "il":

			predictions_of_actions_for_unlabeled = reconstruction_model.predict( X = self.gridworld.standarized_features, y = self.gridworld.policy, unlabeled = unlabeled_indices)
			reconstructed_policy = self.gridworld.policy.copy() ##Use seen actions for states you've seen
			reconstructed_policy[ unlabeled_indices] = predictions_of_actions_for_unlabeled
			inferred_reward_weights = None
		
		elif reconstruct_hyperparams[ "model" ] == "irl":

			reconstructed_policy, inferred_reward_weights, v = reconstruction_model.calculate_policy(	n_actions = self.gridworld.n_actions, transition_probability = self.gridworld.transition_probas,
						       												feature_matrix = self.gridworld.single_state_features, trajectories = reconstruction_trajectories, 
																			terminal_states = self.gridworld.terminal_indices)
			predictions_of_actions_for_unlabeled = reconstructed_policy[unlabeled_indices]
		else:
			assert False , "Only grf, irl, extraction implemented, not " + str( reconstruct_hyperparams[ "model" ] )

		return reconstructed_policy, predictions_of_actions_for_unlabeled, inferred_reward_weights
	