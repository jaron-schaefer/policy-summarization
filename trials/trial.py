import yaml
import os
import yaml
import sys
import ast
import numpy as np
import math
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridworld import Gridworld
import random
import pandas as pd
from experiment_code.constants import TRIALDIR, TESTTRIALDIR
from experiment_code.experiment_utils import get_sim_to_summary, get_performance, get_sim_between_trajectories
from trials.compute_transfer_accurcay import get_transfer_world_policy_accuray
from trials.mirror_trial import mirror_trial

class Trial:
    """This class representsa trial and It is used for generating trials"""

    def __init__(self):
        self.experiment_trial_dir = TRIALDIR

    def get_trial(self, trial_path):
        """Get trial data from a file.

        :param trial_name: The name of the trial.
        :type trial_name: str
        """
        with open(trial_path, 'r') as file:
            trial_data = yaml.safe_load(file)
        
        self.trial_data = trial_data

        self.trial_name = trial_data['TRIAL']
        self.comment = trial_data['COMMENT']

        self.reward_weights = trial_data['REWARDS']
        self.discount = trial_data['DISCOUNT']

        self.summary_method = trial_data['SUMMARY_METHOD']
        self.summary = trial_data['SUMMARY']
        self.extract_hyperparameters = trial_data['EXTRACTION_HYPERPARAMETERS']
        self.reconstruction_hyperparams = trial_data['RECONSTRUCTION_HYPERPARAMETERS']
        self.reconstruction_accuracy_gridworld_1 = trial_data['RECONSTRUCTION_ACCURACY_GRIDWORLD_1']
        self.reconstructed_reward_weights = trial_data['RECONSTRUCTED_REWARD_WEIGHTS']

        self.model_path_gridworld_1 = trial_data['MODEL_PATH_GRIDWORLD_1']
        self.model_reward_gridworld_1 = trial_data['MODEL_REWARD_GRIDWORLD_1']
        self.model_performance_gridworld_1 = trial_data['MODEL_PERFORMANCE_GRIDWORLD_1']

        self.grid_list_1 = trial_data['GRIDWORLD_1']
        self.start_state_gridworld_1 = trial_data['START_STATE_GRIDWORLD_1']
        self.start_state_path_sim = trial_data['START_STATE_PATH_SIMILARITY']
        self.policy_gridworld_1 = trial_data['POLICY_GRIDWORLD_1']
        self.optimal_path_gridworld_1 = trial_data['OPTIMAL_PATH_GRIDWORLD_1']
        self.optimal_reward_gridworld_1 = trial_data['OPTIMAL_REWARD_GRIDWORLD_1']

        self.reconstruction_accuracy_gridworld_2 = trial_data['RECONSTRUCTION_ACCURACY_GRIDWORLD_2']
        self.model_path_gridworld_2 = trial_data['MODEL_PATH_GRIDWORLD_2']
        self.model_reward_gridworld_2 = trial_data['MODEL_REWARD_GRIDWORLD_2']
        self.model_performance_gridworld_2 = trial_data['MODEL_PERFORMANCE_GRIDWORLD_2']

        self.grid_list_2 = trial_data['GRIDWORLD_2']
        self.start_state_gridworld_2 = trial_data['START_STATE_GRIDWORLD_2']
        self.model_path_similarity = trial_data['MODEL_PATH_SIMILARITY']
        self.policy_gridworld_2 = trial_data['POLICY_GRIDWORLD_2']
        self.optimal_path_gridworld_2 = trial_data['OPTIMAL_PATH_GRIDWORLD_2']
        self.optimal_reward_gridworld_2 = trial_data['OPTIMAL_REWARD_GRIDWORLD_2']




    def generate_all_possible_trials(self):
        """Generate trials for every trial world combination"""

        with open('trial_generation/generated_gridworlds.yaml', 'r') as file:
                    all_trial_worlds = yaml.safe_load(file)

        best_summaries_df = pd.read_csv("../data/model_data/best_results.csv")

        for trial in all_trial_worlds:
            
            try:
                il_extract_trial_row = best_summaries_df[best_summaries_df["Name"] == (str(trial)+"_IL_IL.csv")]
                irl_extract_trial_row = best_summaries_df[best_summaries_df["Name"] == (str(trial)+"_IRL_IRL.csv")]
            except Exception as e:
                    print(trial)
                    print(e)
                    continue

            for extraction_method in ["IRL","IL"]:

                if extraction_method == "IRL":
                    trial_row = irl_extract_trial_row
                    other_row = il_extract_trial_row
                else:
                    trial_row = il_extract_trial_row
                    other_row = irl_extract_trial_row


                full_trial_name = str(trial)+"_"+ str(extraction_method)+ '_'+ str(extraction_method)
                discount = 1
                rewards = all_trial_worlds[trial][0]

                # hyperparameter analyisis
                try:
                    summary = ast.literal_eval(trial_row["summary"].values[0])
                    extract_hyperparams = ast.literal_eval(trial_row["extraction_hyperparams"].values[0])
                    reconstruction_hyperparams = ast.literal_eval(trial_row["reconstruction_hyperparams"].values[0])
                    reconstruction_accuracy_gridworld_1 = float(trial_row["accuracy"].values[0])
                    reconstructed_reward_weights = trial_row["inferred_reward_weights"].values[0]
                    if type(reconstructed_reward_weights) == str:
                        reconstructed_reward_weights = np.fromstring(reconstructed_reward_weights[1:-1], sep=' ').tolist()
                except Exception as e:
                    print(trial)
                    print(e)
                    continue

                

                ## Gridworld 1
                grid_list_1 = all_trial_worlds[trial][1]
                first_gridworld = Gridworld(grid_list_1, rewards, discount)
                start_state_gridworld_1, start_state_path_sim = self.find_start_state_girdworld_1(grid_list_1, rewards, summary)
                start_state_1_indice = first_gridworld.state_identifiers.index(start_state_gridworld_1)
                optimal_path_gridworld_1, optimal_reward_gridworld_1 = first_gridworld.get_optimal_path(start_state_1_indice)
                policy_gridworld_1 = first_gridworld.policy.tolist()

                # get model performance grdiworld 1
                reconstructed_policy_gridworld_1 = trial_row["reconstructed_policy"].values[0]
                reconstructed_policy_gridworld_1 = reconstructed_policy_gridworld_1.replace(' ', ',')  
                reconstructed_policy_gridworld_1 = ast.literal_eval(reconstructed_policy_gridworld_1)
                model_path_gridworld_1, model_reward_gridworld_1 = first_gridworld.get_path_to_goal(reconstructed_policy_gridworld_1, start_state_1_indice)
                model_performance_gridworld_1 = get_performance(model_reward_gridworld_1, optimal_reward_gridworld_1)




                ## Gridworld 2
                grid_list_2 = all_trial_worlds[trial][2]
                second_gridworld = Gridworld(grid_list_2, rewards, discount)
                policy_gridworld_2 = second_gridworld.policy.tolist()
                
                try:
                    #get transferworld performance
                    reconstructed_policy_gridworld_2, reconstruction_accuracy_gridworld_2 = get_transfer_world_policy_accuray(first_grid_list=grid_list_1,
                                                                                                                                transfer_grid_list= grid_list_2, 
                                                                                                                                rewards = rewards, 
                                                                                                                                reconstructed_rewards = reconstructed_reward_weights, 
                                                                                                                                summary= summary, 
                                                                                                                                reconstruction_hyperparams = reconstruction_hyperparams)

                    other_reconstructed_reward_weights = other_row["inferred_reward_weights"].values[0] 
                    if type(other_reconstructed_reward_weights) == str:
                        other_reconstructed_reward_weights = np.fromstring(other_reconstructed_reward_weights[1:-1], sep=' ').tolist()
                

                    other_reconstructed_policy_gridworld_2, other_reconstruction_accuracy_gridworld_2 = get_transfer_world_policy_accuray(first_grid_list=grid_list_1,
                                                                                                    transfer_grid_list= grid_list_2, 
                                                                                                    rewards = rewards, 
                                                                                                    reconstructed_rewards = other_reconstructed_reward_weights,
                                                                                                    summary= summary, 
                                                                                                    reconstruction_hyperparams = ast.literal_eval(other_row["reconstruction_hyperparams"].values[0]))

                    start_state_gridworld_2, model_path_sim = self.find_start_state_gridworld_2(grid_list_2, rewards, reconstructed_policy_gridworld_2, other_reconstructed_policy_gridworld_2)
                    
                    start_state_2_indice = second_gridworld.state_identifiers.index(start_state_gridworld_2)
                    
                    # dass nochmal mit other recon policy
                    optimal_path_gridworld_2, optimal_reward_gridworld_2 = second_gridworld.get_optimal_path(start_state_2_indice)
                    model_path_gridworld_2, model_reward_gridworld_2 = second_gridworld.get_path_to_goal(reconstructed_policy_gridworld_2, start_state_2_indice)
                    model_performance_gridworld_2 = get_performance(model_reward_gridworld_2, optimal_reward_gridworld_2)

                except Exception as e:
                    print(trial)
                    print(e)
                    continue


                trial_data = {
                    'TRIAL': str(full_trial_name),
                    'COMMENT': '',
                    'REWARDS': rewards,
                    'DISCOUNT' : discount,

                    'SUMMARY_METHOD': str(extraction_method),
                    'SUMMARY': summary,
                    'EXTRACTION_HYPERPARAMETERS': extract_hyperparams,
                    'RECONSTRUCTION_HYPERPARAMETERS': reconstruction_hyperparams,
                    'RECONSTRUCTION_ACCURACY_GRIDWORLD_1': reconstruction_accuracy_gridworld_1,
                    'RECONSTRUCTED_REWARD_WEIGHTS': reconstructed_reward_weights,

                    'MODEL_PATH_GRIDWORLD_1': model_path_gridworld_1,
                    'MODEL_REWARD_GRIDWORLD_1': model_reward_gridworld_1,
                    'MODEL_PERFORMANCE_GRIDWORLD_1': model_performance_gridworld_1,

                    'GRIDWORLD_1': grid_list_1,
                    'START_STATE_GRIDWORLD_1': start_state_gridworld_1,
                    'START_STATE_PATH_SIMILARITY': start_state_path_sim,
                    'POLICY_GRIDWORLD_1': policy_gridworld_1,
                    'OPTIMAL_PATH_GRIDWORLD_1': optimal_path_gridworld_1,
                    'OPTIMAL_REWARD_GRIDWORLD_1': optimal_reward_gridworld_1,

                    'RECONSTRUCTION_ACCURACY_GRIDWORLD_2': float(reconstruction_accuracy_gridworld_2),
                    'MODEL_PATH_GRIDWORLD_2': model_path_gridworld_2,
                    'MODEL_REWARD_GRIDWORLD_2': model_reward_gridworld_2,
                    'MODEL_PERFORMANCE_GRIDWORLD_2': model_performance_gridworld_2,

                    'GRIDWORLD_2': grid_list_2,
                    'START_STATE_GRIDWORLD_2': start_state_gridworld_2,
                    'MODEL_PATH_SIMILARITY': model_path_sim,
                    'POLICY_GRIDWORLD_2': policy_gridworld_2,
                    'OPTIMAL_PATH_GRIDWORLD_2': optimal_path_gridworld_2,
                    'OPTIMAL_REWARD_GRIDWORLD_2': optimal_reward_gridworld_2
                }

                file_path = 'all_trials/{}.yaml'.format(trial+"_"+ str(extraction_method))

                with open(file_path, 'w') as file:
                    yaml.dump(trial_data, file, default_flow_style=False)

                if extraction_method == "IL":
                    mirror_trial(file_path)
            


    def find_start_state_gridworld_2(self, grid_list, reward_weights, recon_policy, other_recon_policy):
        """_summary_

        :param grid_list: _description_
        :type grid_list: _type_
        :param reward_weights: _description_
        :type reward_weights: _type_
        :param summary: _description_, defaults to None
        :type summary: _type_, optional
        :return: _description_
        :rtype: _type_
        """

        gridworld = Gridworld(grid_list, reward_weights)
        valid_states = gridworld.state_identifiers.copy()
        # valid states in ascending order, represented as state identifiers
        valid_states = self.sort_start_states_by_distance(valid_states, gridworld)

        # remove states closer than 3 states to goal states (also goal states)
        for state in valid_states.copy():
            for goal_state_indice in gridworld.terminal_indices:
                if len(valid_states) == 1:
                    return valid_states[0], 2 
                if self.distance_between_states(state, gridworld.state_identifiers[goal_state_indice]) < 3:
                    if state in valid_states:
                        valid_states.remove(state)

        # find start states where the optimal path does not only contain stone fields
        for state in valid_states.copy():

            state_indice = gridworld.state_identifiers.index(state)
            optimal_path, optimal_reward = gridworld.get_optimal_path(state_indice)
            path_crosses_feature = False

            for i, [stat, action] in enumerate(optimal_path):
                
                if i == 0:
                    continue

                if gridworld.single_state_features[stat].tolist() != [1,0,0,0,0,0,0,0,0]:
                    path_crosses_feature = True
                    break

            if len(valid_states) == 1:
                return valid_states[0], 2
            
            if not path_crosses_feature:
                valid_states.remove(state)

        # # find start states where the optimal path is not a straigth line (always taking the same action)
        for state in valid_states.copy():

            state_indice = gridworld.state_identifiers.index(state)
            optimal_path, optimal_reward = gridworld.get_optimal_path(state_indice)
            first_action = optimal_path[0][1]
            straigth_path = True

            for _, action in optimal_path:
                
                if action != first_action:
                    straigth_path = False
                    break

            if len(valid_states) == 1:
                return valid_states[0], 2 
            
            if straigth_path:
                valid_states.remove(state)

        lowest_similarity = float('inf')
        start_state_with_lowest_similarity = None

        for possible_start_state in valid_states:
            start_state_indice = gridworld.state_identifiers.index(possible_start_state)

            il_path, il_reward = gridworld.get_path_to_goal(recon_policy.tolist(), start_state_indice)
            irl_path, irl_reward = gridworld.get_path_to_goal(other_recon_policy.tolist(), start_state_indice)

            if len(il_path) == 50 or len(irl_path) == 50:  # means no path found (probably in loop)
                continue
            
            similarity = get_sim_between_trajectories(il_path, irl_path)

            if similarity < lowest_similarity:
                lowest_similarity = similarity
                start_state_with_lowest_similarity = possible_start_state
        
        if start_state_with_lowest_similarity == None:
            return valid_states[-1], 2
        
        return start_state_with_lowest_similarity, lowest_similarity



            
    def find_start_state_girdworld_1(self, grid_list, reward_weights, summary):
        """_summary_

        :param grid_list: _description_
        :type grid_list: _type_
        :param reward_weights: _description_
        :type reward_weights: _type_
        :param summary: _description_, defaults to None
        :type summary: _type_, optional
        :return: _description_
        :rtype: _type_
        """

        gridworld = Gridworld(grid_list, reward_weights)
        valid_states = gridworld.state_identifiers.copy()
        # valid states in ascending order, represented as state identifiers
        valid_states = self.sort_start_states_by_distance(valid_states, gridworld)

        # remove states closer than 3 states to goal states (also goal states)
        for state in valid_states.copy():
            for goal_state_indice in gridworld.terminal_indices:
                if len(valid_states) == 1:
                    return valid_states[0], 2 
                if self.distance_between_states(state, gridworld.state_identifiers[goal_state_indice]) < 3:
                    if state in valid_states:
                        valid_states.remove(state)


        # remove all states on a trajectory 
        states_in_summary = []
        for sum_trajectory in summary:
            for state_indice, _ in sum_trajectory:
                states_in_summary.append(gridworld.state_identifiers[state_indice])
        
        for state in valid_states.copy():
            # if no states left return the state that is the farthest away
            if len(valid_states) == 1 and state in states_in_summary:
                return valid_states[0], 2
            if state in states_in_summary:
                valid_states.remove(state)


        # find start states where the optimal path does not only contain stone fields
        for state in valid_states.copy():

            state_indice = gridworld.state_identifiers.index(state)
            optimal_path, optimal_reward = gridworld.get_optimal_path(state_indice)
            path_crosses_feature = False

            for i, [stat, action] in enumerate(optimal_path):
                
                if i == 0:
                    continue

                if gridworld.single_state_features[stat].tolist() != [1,0,0,0,0,0,0,0,0]:
                    path_crosses_feature = True
                    break

            if len(valid_states) == 1:
                return valid_states[0], 2 + get_sim_to_summary(optimal_path, summary)
            
            if not path_crosses_feature:
                valid_states.remove(state)

        # find start states where the optimal path is not a straigth line (always taking the same action)
        for state in valid_states.copy():

            state_indice = gridworld.state_identifiers.index(state)
            optimal_path, optimal_reward = gridworld.get_optimal_path(state_indice)
            first_action = optimal_path[0][1]
            straigth_path = True

            for _, action in optimal_path:
                
                if action != first_action:
                    straigth_path = False
                    break

            if len(valid_states) == 1:
                return valid_states[0], 1 + get_sim_to_summary(optimal_path, summary)
            
            if straigth_path:
                valid_states.remove(state)


        # find state which optimal trajectory has the lowest similarity to summary trajectories 
        valid_states.reverse()# so state with the farthest distance is the first
        similarities = {}
        for state in valid_states.copy():

            state_indice = gridworld.state_identifiers.index(state)
            optimal_path, optimal_reward = gridworld.get_optimal_path(state_indice)
            state_tuple = tuple(state)

            # get similarity to summary trajectories
            similarities[state_tuple] = get_sim_to_summary(optimal_path, summary)
        
        # Sort the keys based on two criteria: the value of the key and its position in the dictionary
        sorted_start_states = sorted(similarities, key=lambda x: (similarities[x], list(similarities.keys()).index(x)))

        optimal_start_state = list(map(int, sorted_start_states[0]))

        return optimal_start_state, similarities[sorted_start_states[0]]

   
    
    def sort_start_states_by_distance(self, valid_start_states, gridworld: Gridworld):
        """_summary_

        :param valid_start_states: _description_
        :type valid_start_states: _type_
        :param gridworld: _description_
        :type gridworld: PlayGridworld
        :return: _description_
        :rtype: _type_
        """
        goal_states = [gridworld.state_identifiers[index] for index in gridworld.terminal_indices]
        distances = [
            max(abs(start_state[0] - goal_state[0]), abs(start_state[1] - goal_state[1]))
            for start_state in valid_start_states
            for goal_state in goal_states
        ]
        sorted_start_states = [state for _, state in sorted(zip(distances, valid_start_states))]
        return sorted_start_states

    def distance_between_states(self, state1, state2):
        return max(abs(state1[0] - state2[0]), abs(state1[1] - state2[1]))




    def make_experiment_trials(self, number_of_trials = 32):
        """_summary_

        :param number_of_trials: _description_
        :type number_of_trials: _type_
        """

        best_trials = self.find_trials_with_lowest_similarity_start_states()

        print(best_trials)

        for i, trial_variation in enumerate(best_trials):
            
            for extraction_method in ["IL","IRL"]:

                var_comb = trial_variation.split("_")[1:]
                var_comb = "_".join(var_comb)

                new_trial_name = str(int(i%2)) + "_" + var_comb + "_" + extraction_method
                
                old_trial_path = os.path.join('all_trials', str(trial_variation)+ "_" + extraction_method+ ".yaml")

                with open(old_trial_path, 'r') as file:
                    yaml_content = yaml.safe_load(file)

                print(yaml_content["TRIAL"] + ' --->'+ new_trial_name)

                yaml_content["OLD_TRIAL_NAME"] = yaml_content["TRIAL"]
                yaml_content["TRIAL"] = new_trial_name

                with open(os.path.join(TRIALDIR, new_trial_name+ ".yaml"), 'w') as file:
                    yaml.safe_dump(yaml_content, file)

                    
        print(str(len(best_trials)*2) + ' trials are saved!')


    def find_trials_with_lowest_similarity_start_states(self):

        #TODO: Add more constraints
    
        # start state of transfer world should also be as optimal as possible (lowest similarity)
        # start state of first gridworld best choice



        # make df with name and similarity
        trials_with_sim = []
        # Iterate over all trials
        for trial_name in os.listdir('all_trials'):
            file_path = os.path.join('all_trials', trial_name)

            # Open the YAML file and load its contents
            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)

            # first world performance needs to be better than 0
            # second world performance needs to be better than 0
            if (yaml_data.get('MODEL_PERFORMANCE_GRIDWORLD_1') == 0):
                continue

            # Extract the value at the desired key
            start_state_path_sim = yaml_data.get('START_STATE_PATH_SIMILARITY')
            model_path_sim = yaml_data.get('MODEL_PATH_SIMILARITY')
            trials_with_sim.append({'trial_name': trial_name, 'sim': start_state_path_sim + model_path_sim})

        trials_with_sim_df = pd.DataFrame(trials_with_sim)  
        

        with open('trial_generation/variation_combinations.yaml', 'r') as file:
            all_variation_combinations = yaml.safe_load(file)

        variations_with_sim = []
        for var_comb in all_variation_combinations:
            filtered_df = trials_with_sim_df[trials_with_sim_df['trial_name'].str.contains(str(var_comb))]

            filtered_df['trial_number'] = filtered_df['trial_name'].str.split('_').str[0]

            grouped_df = filtered_df.groupby('trial_number')
            groups_with_size_2 = grouped_df.filter(lambda x: len(x) == 2) # only include trials with to working summary for IL and IRL
            result = groups_with_size_2.groupby('trial_number')['sim'].sum()

            if len(result) < 2:
                print(var_comb)

            best_trials = result.nsmallest(2)
            
            for trial_number, sim in best_trials.items():
                variations_with_sim.append(str(trial_number) +"_" + str(var_comb))
        
        return variations_with_sim
    


    def make_trial_df(self, path):

        trial_data=[]
        
        for trial_name in os.listdir(path):
            trial = Trial()
            trial.get_trial(os.path.join(path, trial_name))


            trial_data.append({
                'trial_name': trial.trial_name,
                "density": trial.trial_name.split("_")[1],
                "nfeatures": trial.trial_name.split("_")[2],
                "ndistractions": trial.trial_name.split("_")[3],
                "extraction_method": trial.trial_name.split("_")[4],
                "condition": "1",
                'reward_weights': trial.reward_weights,
                'discount': trial.discount,
                'summary_method': trial.summary_method,
                'summary': trial.summary,
                'extract_hyperparameters': trial.extract_hyperparameters,
                'reconstruction_hyperparams': trial.reconstruction_hyperparams,

                'reconstruction_accuracy': trial.reconstruction_accuracy_gridworld_1,
                'reconstructed_reward_weights': trial.reconstructed_reward_weights,
                
                'model_path': trial.model_path_gridworld_1,
                'model_reward': trial.model_reward_gridworld_1,
                'model_performance': trial.model_performance_gridworld_1,
                'grid_list': trial.grid_list_1,
                'start_state': trial.start_state_gridworld_1,
                'start_state_path_sim': trial.start_state_path_sim,
                'model_path_similarity': None,
                'policy': trial.policy_gridworld_1,
                'optimal_path': trial.optimal_path_gridworld_1,
                'optimal_reward': trial.optimal_reward_gridworld_1,
                'percent_of_optimal_reward': max(0, (int(trial.model_reward_gridworld_1) / int(trial.optimal_reward_gridworld_1)))})
            
            trial_data.append({
                'trial_name': trial.trial_name,
                "density": trial.trial_name.split("_")[1],
                "nfeatures": trial.trial_name.split("_")[2],
                "ndistractions": trial.trial_name.split("_")[3],
                "extraction_method": trial.trial_name.split("_")[4],
                "condition": "2",
                'reward_weights': trial.reward_weights,
                'discount': trial.discount,
                'summary_method': trial.summary_method,
                'summary': trial.summary,
                'extract_hyperparameters': trial.extract_hyperparameters,
                'reconstruction_hyperparams': trial.reconstruction_hyperparams,

                'reconstruction_accuracy': trial.reconstruction_accuracy_gridworld_2,
                'reconstructed_reward_weights': trial.reconstructed_reward_weights,
                
                'model_path': trial.model_path_gridworld_2,
                'model_reward': trial.model_reward_gridworld_2,
                'model_performance': trial.model_performance_gridworld_2,
                'grid_list': trial.grid_list_2,
                'start_state': trial.start_state_gridworld_2,
                'start_state_path_sim': None,
                'model_path_similarity': trial.model_path_similarity,
                'policy': trial.policy_gridworld_2,
                'optimal_path': trial.optimal_path_gridworld_2,
                'optimal_reward': trial.optimal_reward_gridworld_2,
                'percent_of_optimal_reward': max(0, (int(trial.model_reward_gridworld_2) / int(trial.optimal_reward_gridworld_2)))})


  
            
        df = pd.DataFrame(trial_data)
        # Print the resulting DataFrame
        df.to_csv('df_all_' +str(path)+'.csv', index=False)