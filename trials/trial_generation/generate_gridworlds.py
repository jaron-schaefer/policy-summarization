import random
import yaml
import numpy as np
import itertools

class GridworldGenerator:

    def generate_gridworlds(self, max_x, max_y, trial):
        """
        Generates rewards and two random gridworlds with specified parameters.

        :param max_x: The maximum x-coordinate value of the grid.
        :type max_x: int
        :param max_y: The maximum y-coordinate value of the grid.
        :type max_y: int
        :param trial: The trial name indicating the configuration of the gridworlds.
        :type trial: str
        :return: A tuple containing the rewards, gridworld 1, and gridworld 2.
        :rtype: tuple
        """

        # number of features (plus one for terminal states and one for '-')
        non_terminal_features = ['B','C','D','E','F','G','H']
        included_features = ['-','A']

        # gridworld density
        if 'HD' in trial:
            gridworld_density = self._generate_gaussian(0.75, 0.05)
        elif 'LD' in trial:
            gridworld_density = self._generate_gaussian(0.4, 0.05)
        else:
            assert False, " HD or LD wasnt included in trial name"

        # state dimensions (amount of single state features)
        weights = [0.25, 0.5, 0.25]
        if 'HF' in trial:
            n_features = random.choices([7, 8, 9], weights=weights, k=1)[0]
            non_terminal_included_features = non_terminal_features[:n_features]  
        elif 'LF' in trial:
            n_features = random.choices([4, 5, 6], weights=weights, k=1)[0]
            non_terminal_included_features = non_terminal_features[:n_features]
        else:
            assert False, "HF or LF wasnt included in trial name"
        included_features = included_features + non_terminal_included_features
        
        # distraction states
        if 'HS' in trial:
            dis_prob = 0.75
        elif 'LS' in trial:
            dis_prob = 0
        else:
            assert False, " HS or LS wasnt included in trial name"

        # generate rewards
        rewards = [-1,100,0,0,0,0,0,0,0]

        #make rewards (and distraction features)
        for i in range(2,len(rewards)):
            random_reward = -1
            while random_reward in rewards:
                random_reward = random.randint(-20, -2)
            rewards[i] = -1 if random.random() < dis_prob else random_reward

        gridworld_1 = self._make_random_gridworld(max_x, max_y, gridworld_density, included_features)
        gridworld_2 = self._make_random_gridworld(max_x, max_y, gridworld_density, included_features)
        
        return rewards, gridworld_1, gridworld_2


    def _make_random_gridworld(self, max_x, max_y, gridworld_density, included_features):
        """
        Generates a random gridworld with specified parameters.

        :param max_x: The maximum x-coordinate value of the grid.
        :type max_x: int
        :param max_y: The maximum y-coordinate value of the grid.
        :type max_y: int
        :param gridworld_density: The density of the gridworld, indicating the probability of each state being occupied.
        :type gridworld_density: float
        :param included_features: The list of features to be included in the gridworld.
        :type included_features: list[str]
        :return: The generated gridworld represented as a list of strings.
        :rtype: list[str]
        """
        gridworld = [['-' for _ in range(max_x)] for _ in range(max_y)]

        n_states = max_x * max_y

        feature_amounts = self._make_feature_amounts(n_states, gridworld_density, included_features)

        # place features
        for feature in feature_amounts.keys():
            count = feature_amounts[feature]  # Anzahl der Vorkommen des Features
            
            while count > 0:

                row = random.randint(0, max_y - 1)
                col = random.randint(0, max_x - 1)
                
                # check that the selected state is empty
                if gridworld[row][col] == '-':
                    gridworld[row][col] = feature
                    count -= 1

        #change representation 
        final_gridworld = []
        for row in gridworld:
            transformed_row = ' '.join(row)
            final_gridworld.append(transformed_row)

        return final_gridworld


    def _make_feature_amounts(self, n_states, gridworld_density, included_features):
        """
        Determines the amount of each feature to be included in the gridworld.

        :param n_states: The total number of states in the gridworld.
        :type n_states: int
        :param gridworld_density: The density of the gridworld, indicating the probability of each state being occupied.
        :type gridworld_density: float
        :param included_features: The list of features to be included in the gridworld. (wirh out normal states "-")
        :type included_features: list[str]
        :return: A dictionary mapping each feature to its corresponding amount in the gridworld.
        :rtype: dict
        """
        assert included_features[0] == '-'
        assert included_features[1] == 'A'
        included_features  = included_features[2:]

        n_filled = int(n_states * gridworld_density)
        n_A = random.choice([1,2])

        dict = {'-':n_states - n_filled,'A':n_A,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0}
        
        for f in included_features:
            dict[f] = 1

        count = n_A  + len(included_features)

        while n_filled - count > 0: 

            for f in included_features:

                if n_filled - count > 4:
                    x = random.randint(0, int((n_filled - count)/4))
                else:
                    x = random.randint(0, n_filled - count)

                dict[f] += x
                count += x

        return dict


    def _generate_gaussian(self,mu, v):
        """
        Generates a random value from a Gaussian distribution with specified mean and variance.

        :param mu: The mean of the Gaussian distribution.
        :type mu: float
        :param v: The variance of the Gaussian distribution.
        :type v: float
        :return: A randomly generated value within the specified range.
        :rtype: float
        """
        while True:
            value = random.gauss(mu, v)
            if value >= 0.2 and value <= 0.95:
                return value




def save_variation_combinations():

    combinations = list(itertools.product(['HD', 'LD'], ['HF', 'LF'], ['HS','LS']))

    all_variations = ['_'.join(combination) for combination in combinations]
    print(len(all_variations))

    with open('variation_combinations.yaml', 'w') as file:
        yaml.dump(all_variations, file, default_flow_style=False)
        print("Saved all variation combinations")

#save_variation_combinations()



def make_gridworlds_for_all_variations(n_gridworlds_per_variation):
    gridworld_generator = GridworldGenerator()
    with open('variation_combinations.yaml', 'r') as file:
            all_variations = yaml.safe_load(file)

    data = {}
    for i in range(n_gridworlds_per_variation):
        for trial in all_variations:
            rewards, gridworld_1, gridworld_2 = gridworld_generator.generate_gridworlds(7, 5, trial)
            data[str(i) +'_'+ trial] = [rewards, gridworld_1, gridworld_2]

    file_path = 'generated_gridworlds.yaml'

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
        print("Saved all generated gridworlds")


#make_gridworlds_for_all_variations(50)