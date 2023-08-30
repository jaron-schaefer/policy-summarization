import yaml
import sys
import os
import warnings
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.hyperparameter_comparison_pipeline import HyperparameterComparisonPipeline
from sklearn.model_selection import ParameterGrid

#warnings.filterwarnings("ignore", category=RuntimeWarning)


gammas = [0.1, 1.] 
degrees = [2, 3]
grf_params =   ParameterGrid( [ { "kernel": [ "rbf" ] , "gamma": gammas },{ "kernel": [ "poly" ] , "gamma": gammas , "degree": degrees } ] )

maxent_params = ParameterGrid( [ {"epochs": [300] ,"discount":[0.99], "learning_rate": [0.5, 1.] , "gd_threshold": [ 1e-5 ] ,"mc_rollouts": [0] , 
                                    "rollout_horizon": [20], "par": [False] } ] )

summary_length = [5]
irl_extraction_hyperparams =[{ "model": ["irl"] ,"algorithm": ["extraction"],"summary_length":summary_length, "to_make_multiple_summaries": [1,2,3]}]

il_extraction_hyperparams = ([{"model": ["il"], "algorithm": ["extraction"],"summary_length":summary_length, "strategy": ["al"], "kernel_params": [params]} for params in grf_params])



irl_reconstruction_hyperparams = [ { "model": ["irl"]  ,"algorithm":  ["reconstruction"]  , "maxent_params": [params]} for params in maxent_params]

il_reconstruction_hyperparams = [ { "model": ["il"] , "algorithm":  ["reconstruction"]   , "kernel_params": [params] } for params in grf_params]


with open("generated_gridworlds.yaml", 'r') as file:
    all_trial_worlds = yaml.safe_load(file)


for trial in tqdm(all_trial_worlds):

    rewards = all_trial_worlds[trial][0]
    grid_list_1 = all_trial_worlds[trial][1]

    gridworld_hyperparams = [{ "grid_list": [grid_list_1.copy()] ,"reward_weights": [rewards.copy()], "trajectory_length": [15], "discount": [1], "rollout_horizon": [20]}]

    pipeline = HyperparameterComparisonPipeline(
                gridworld_hyperparams = gridworld_hyperparams,
                il_extraction_hyperparams= il_extraction_hyperparams,
                irl_extraction_hyperparams = irl_extraction_hyperparams,
                il_reconstruction_hyperparams = il_reconstruction_hyperparams,
                irl_reconstruction_hyperparams = irl_reconstruction_hyperparams)
    
    pipeline.compute_reconstruction_accuracies(trial)

    