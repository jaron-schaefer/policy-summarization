from models.modelling_gridworld import ModelGridworld
from sklearn.metrics import accuracy_score
from models.reconstruction.il_grf import GRF
import numpy as np


def get_transfer_world_policy_accuray(first_grid_list, transfer_grid_list, rewards, reconstructed_rewards, summary, reconstruction_hyperparams):

    optimal_transfer_gridworld = ModelGridworld(transfer_grid_list, rewards)


    if reconstruction_hyperparams[ "model" ] == "irl":

        recon_gridworld = ModelGridworld(transfer_grid_list,  reconstructed_rewards)

        accuracy = accuracy_score( optimal_transfer_gridworld.policy , recon_gridworld.policy)

        return recon_gridworld.policy, accuracy
    
    elif reconstruction_hyperparams[ "model" ] == "il":

        first_gridworld = ModelGridworld(first_grid_list, rewards)

        reconstruction_model = GRF( reconstruction_hyperparams[ "kernel_params" ] , n_classes=optimal_transfer_gridworld.n_actions )

        X = optimal_transfer_gridworld.standarized_features.tolist() # standarized features of transfer world + standarized features for summary states (of first gridworld)
        
        y = optimal_transfer_gridworld.policy.tolist() # policy of transfer world + optimal actions for states in summary
        for traj in summary:

            for state, action in traj:
                if state in first_gridworld.terminal_indices:
                    continue
                y.append(action)
                X.append(first_gridworld.standarized_features[state].tolist())

        unlabeled_indices = [x for x in range(len(optimal_transfer_gridworld.policy))] # whole transfer world
        predictions_of_actions_for_unlabeled = reconstruction_model.predict( X = np.array(X), y = np.array(y), unlabeled = np.array(unlabeled_indices))

        reconstructed_policy = predictions_of_actions_for_unlabeled # since whole world was unlabled

        accuracy = accuracy_score( optimal_transfer_gridworld.policy , reconstructed_policy)
		
        return reconstructed_policy, accuracy
    else:
        assert False , "Only il, irl, extraction implemented, not " + str( reconstruction_hyperparams[ "model" ] )
    