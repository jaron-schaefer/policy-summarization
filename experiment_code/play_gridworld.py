import os
import sys
import pygame
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gridworld import Gridworld
from experiment_code.constants import *


class PlayGridworld(Gridworld):

    def __init__(self, grid_list, reward_weights, screen = None, display = None, discount = 1):
        super().__init__(grid_list, reward_weights, discount)
        self.screen = screen
        self.display = display

    def step(self, action):
        self.total_path.append([self.agent_state, action])
        next_state = self.get_next_state(self.agent_state , action)
        self.agent_state = next_state

        reward = self.rewards[self.agent_state]
        self.total_reward += reward

        done = self.agent_state in self.terminal_indices
        self.update_gridworld_view(self.state_identifiers[next_state])
        return next_state, reward, done

    
    def reset(self, start_state, feature_imgs, summary = None, summary_grid_list = None):
        self.make_gridworld_gamefield(feature_imgs, summary, summary_grid_list)
        self.start_state = self.state_identifiers.index(start_state)
        self.agent_state = self.start_state
        self.optimal_path, self.optimal_reward = self.get_optimal_path(self.start_state)
        self.total_reward = 0
        self.total_path = []
        self.update_gridworld_view(start_state)


    # --------- Methods for drawing Gridworld -----------------

    def draw_features(self, square_size, surface):

        for i, feature in enumerate(self.single_state_features):
            feature = feature.tolist()
            landscape = self.feature_imgs[feature.index(1)]
            path = os.path.join(LANDSCAPESDIR,landscape)
            state_x, state_y = self.state_identifiers[i]
            x, y = self.transform_state_ident_to_coordinates(square_size, state_x, state_y)
            image = pygame.image.load(path) 
            image = pygame.transform.scale(image, (square_size, square_size))
            surface.blit(image, (x, y))

    def draw_gridlines(self, square_size, surface):
        # Draw vertical lines
        for i in range(self.max_x + 1):
            x = i * square_size
            pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, self.max_y * square_size), width=3)

        # Draw horizontal lines
        for j in range(self.max_y + 1):
            y = j * square_size
            pygame.draw.line(surface, (0, 0, 0), (0, y), (self.max_x * square_size, y), width=3)

    def gridworld_surface(self, square_size):
        gridworld_surface = pygame.Surface((self.max_x * square_size, self.max_y * square_size))
        self.draw_features(square_size, gridworld_surface)
        self.draw_gridlines(square_size, gridworld_surface)
        return gridworld_surface

    def summary_surface(self, square_size, summary, summary_grid_list = None, bgc = BGC):

        if summary_grid_list is not None:
            summary_world = PlayGridworld(summary_grid_list, self.reward_weights, self.screen, self.display)
            summary_world.feature_imgs = self.feature_imgs   
            summary_surface = summary_world.summary_surface(square_size, summary, bgc = (90, 90, 90, 255) )
            return summary_surface
        
        summary_surface = pygame.Surface((LENGTH_SUMMARY_SURFACE, HEIGHT_SUMMARY_SURFACE)) 
        summary_surface.fill(bgc)    
        grid_x = SUMMARY_SQUARE_SIZE
        for i, trajectory in enumerate(summary):
            gridworld_surface = self.gridworld_surface(square_size)
            self.draw_trajectory(square_size, trajectory, gridworld_surface)
            summary_surface.blit(gridworld_surface, (grid_x, SUMMARY_SQUARE_SIZE))
            grid_x += self.max_x*square_size + SUMMARY_GAP_SIZE
        return summary_surface
            
    def transform_state_ident_to_coordinates(self, square_size, x, y):
        #transform gridworld coordinates to pygame coordinates
        pygame_x = x * square_size
        pygame_y = (self.max_y - y - 1) * square_size
        return pygame_x, pygame_y

    def draw_trajectory(self, square_size, trajectory, surface):

        for i, [state, action] in enumerate(trajectory):
            if state in self.terminal_indices:
                break
            if i == 0:
                arrow_image_path = os.path.join(STARTARROWSDIR, str(action) +'.png')
            else:
                arrow_image_path = os.path.join(ARROWSDIR, str(action) +'.png')
            state_x, state_y = self.state_identifiers[state]
            x, y = self.transform_state_ident_to_coordinates(square_size, state_x, state_y)
            image = pygame.image.load(arrow_image_path) 
            image = pygame.transform.scale(image, (square_size, square_size))
            surface.blit(image, (x, y))

    def make_gridworld_gamefield(self, feature_imgs, summary, summary_grid_list=None):
        background = pygame.Surface((screen_width, screen_height))
        background.fill(BGC)
    
        self.feature_imgs = feature_imgs
        gridworld = self.gridworld_surface(SQUARE_SIZE)
        background.blit(gridworld, (GRID_X, GRID_Y))

        summary = self.summary_surface(SUMMARY_SQUARE_SIZE, summary, summary_grid_list)
        background.blit(summary, SUMMARY_POS)
        
        self.background = background
        
    

    def update_gridworld_view(self, state):
        x,y = self.transform_state_ident_to_coordinates(SQUARE_SIZE, state[0], state[1])
        x = x + GRID_X
        y = y + GRID_Y
        agent = pygame.image.load(AGENTIMGPATH) 
        transformed_agent = pygame.transform.scale(agent, (SQUARE_SIZE, SQUARE_SIZE))
        
        background = self.background.copy()
        background.blit(transformed_agent, (x, y))
        
        self.screen.clear()
        self.screen.screen.blit(background,(0,0))
        self.display.fill(screen=self.screen) 
        self.display.show()
        pygame.image.save(background, "screenshot.png")



# grid_list = ['A B C D E F G',
#              'H - - - - - -',
#              '- - - - - - -',
#              '- - - - - - -',
#              '- - - - - - -']

# rewards = [-1,15,-1,-1,-1,-1,-1,-1,-1]
# trajectories = [[[1,0],[2,1],[3,2],[3,2]],[[1,0],[2,1],[3,2],[3,2]],[[1,0],[2,1],[3,2],[3,2]],[[1,0],[2,1],[3,2],[3,2]],[[1,0],[2,1],[3,2],[3,2]]]
# features_imgs = sorted(os.listdir(LANDSCAPESDIR))
# gridworld = PlayGridworld(grid_list, rewards)
# gridworld.save_gridworld_image(features_imgs, trajectories)
