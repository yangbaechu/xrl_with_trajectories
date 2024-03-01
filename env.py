import logging
import random

import numpy as np
import warnings

from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns

from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)

ACTION_DICT = {0: 'LEFT', 1: 'UP', 2: 'RIGHT', 3: 'DOWN'}


# Utility functions for locating states in the environment
def coords_to_idx(row, col, grid_dim=(6, 9)):
    """Get 0-53 index from given co-ordinates"""
    return row * grid_dim[1] + col


def idx_to_coords(idx, grid_dim=(6, 9)):
    """Get co-ordinates from given index"""
    return idx // grid_dim[1], idx % grid_dim[1]


class Environment:
    """
        Environment Implementation
    """

    def __init__(self,
                 dim=(6, 9),
                 starts=[(0, 0)],
                 terminals={(0, 8): +1, (5, 8): -1},
                 obstacles=[[0, 7], [1, 7], [2, 7], [1, 1], [2, 1], [3, 2], [4, 5]],
                 stoc=0,
                 random_start=False):
        
        self.dim = dim

        # Set of start, end and obstacle states
        if not random_start:
            self.starts = np.array(starts)
        else:
            self.starts = []
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    if (i,j) not in terminals and [i,j] not in obstacles:
                        self.starts.append((i,j))
            self.starts = np.array(self.starts)
        self.terminals = terminals
        self.obstacles = obstacles

        # Rewards
        self.rewards = np.ones(shape=dim) * (-0.1)

        # Add terminations along with corresponding rewards
        for coords, reward in terminals.items():
            self.rewards[coords] = reward

        # Stochasticity in the environment
        self.stoc = stoc

        # Set player position to one of the states in the start set
        self.player_pos = random.sample(list(self.starts), 1)[0]

        # Action space
        self.action_space = [0, 1, 2, 3]
        
    def plot_traj_densities_with_starts(self, trajectories_lists, starts_lists):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()  # Flatten to make iteration easier

        cmap_ = colors.LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

        for idx, (trajectories, starts) in enumerate(zip(trajectories_lists, starts_lists)):
            canvas = np.zeros(shape=self.dim)
            for traj in trajectories:
                for sars in traj:
                    state = sars[0]
                    i, j = self.idx_to_coords(state)
                    canvas[i, j] += 1

            max_count = np.max(canvas)
            if max_count > 0:
                normalized_canvas = canvas / max_count
            else:
                normalized_canvas = canvas

            im = axs[idx].imshow(normalized_canvas, interpolation='none', aspect='equal', cmap=cmap_)
            axs[idx].set_title(f'Trajectory Group {idx+1}')

            # Overlay starting points
            max_start = np.max(starts)
            if max_start > 0:
                normalized_starts = starts / max_start
            else:
                normalized_starts = starts

            for i in range(starts.shape[0]):
                for j in range(starts.shape[1]):
                    if starts[i, j] > 0:
                        # Overlay a blue dot for starting points, intensity based on count
                        axs[idx].plot(j, i, 'o', color='blue', markersize=normalized_starts[i, j] * 10 + 5, alpha=0.5)

            # Same plotting adjustments as before...

        plt.tight_layout()
        fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
        plt.show()

    def reset(self, set_state=None):
        """Resetting to a given set_state"""

        # Note that the player position is a 2D tuple
        if set_state:
            self.player_pos = set_state

        else:
            self.player_pos = random.sample(list(self.starts), 1)[0]

        # Check if the player position is same as the terminal one
        self.check_done()

        return self.player_pos

    def step(self, action):
        """Performing given action"""

        # Validate initialization of the environment
        if self.player_pos is None:
            warnings.warn('Resetting the environment because it was not done so post initialization...')
            self.reset()

        # Apply stochasticity to choose the action (Epistemic Uncertainty)
        if np.random.random() <= self.stoc:
            action = np.random.choice(self.action_space)
            
        # action = action if (np.random.random() > self.stoc) else np.random.choice(self.action_space)

        # The environment coordinates look as below
        # (0, 0) (0, 1) ...
        # (1, 0) ...

        logging.info(f'ORIG POS:{self.player_pos}')
        if action == 1:  # up
            # UP --> X will change by -1, Y coordinate will remain the same.
            logging.info('UP')
            new_player_pos = (self.player_pos[0] - 1, self.player_pos[1])

        elif action == 0:  # left
            # LEFT --> Y coordinate will change by -1, the X will not change.
            logging.info('LEFT')
            new_player_pos = (self.player_pos[0], self.player_pos[1] - 1)

        elif action == 2:  # right
            # RIGHT --> Y will change by +1, the X will not change
            logging.info('RIGHT')
            new_player_pos = (self.player_pos[0], self.player_pos[1] + 1)

        else:  # down
            # DOWN --> X will change by +1, Y will not change
            logging.info('DOWN')
            new_player_pos = (self.player_pos[0] + 1, self.player_pos[1])

        logging.info(f'PROP MOD POS:{new_player_pos}')

        # Check if the new player position keeps the agent on board. Also check for any obstacle hit.
        # Invalid trasitions should be reverted back to the original position.
        if (new_player_pos[0] >= 0) and (new_player_pos[1] >= 0) and (new_player_pos[0] < self.dim[0]) \
                and (new_player_pos[1] < self.dim[1]) and (list(new_player_pos) not in self.obstacles):
            self.player_pos = new_player_pos

        logging.info(f'NEW POS:{self.player_pos}')
        self.check_done()

        return self.player_pos, self.rewards[tuple(self.player_pos)], self.done, {}

    def check_done(self):
        """Check if terminal state is reached"""
        if tuple(self.player_pos) in self.terminals:
            self.done = True  # Set the done to True
        else:
            self.done = False
        return self.done  # If needed, the value of done can also be extracted from here

    def render(self, title=''):
        
    
#         canvas = np.zeros(shape=self.dim)

        
#         # Mark player position
# #         canvas[tuple(self.player_pos)] = 0.5
        
#         # Add start states
#         for start in self.starts:
#             canvas[tuple(start)] = +0.25

#         # Add end states
#         for coord, reward in self.terminals.items():
#             canvas[coord] = reward

#         # Add obstacles
#         for obstacle in self.obstacles:
#             canvas[tuple(obstacle)] = -0.25

#         plt.figure(figsize=(2,2))
#         plt.title(title)
#         plt.imshow(canvas)
#         plt.show()

        cmap_ = colors.ListedColormap(['black', 'gray', 'lime', 'red']) 
    # 0 - bg
    # 1 - wall
    # 2 - goal
    # 3 - pit
    # 4 - agent
    
        canvas = np.zeros(shape=self.dim)

        # Add end states
        for coord, reward in self.terminals.items():
            if reward > 0:
                canvas[coord] = 2
            elif reward < 0:
                canvas[coord] = 3
        
        
        # Add obstacles
        for obstacle in self.obstacles:
            canvas[tuple(obstacle)] = 1
        
        # Add agent
#         canvas[tuple(self.player_pos)] = 4
            
        
        plt.figure(figsize=(2, 2))
        im = plt.imshow(canvas,
                        interpolation='none', aspect='equal', cmap=cmap_)

        ax = plt.gca();

        # Major ticks
        ax.set_xticks(np.arange(0, self.dim[0], 1))
        ax.set_yticks(np.arange(0, self.dim[1], 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, self.dim[0] + 1, 1))
        ax.set_yticklabels(np.arange(1, self.dim[1] + 1, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.dim[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.dim[1], 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        plt.show()
                    
    
    def idx_to_coords(self, state):
        """
        Converts a state index into grid coordinates.

        Args:
        - state: An integer representing the state index or a tuple already in the form of grid coordinates.

        Returns:
        - A tuple of (i, j) representing the grid coordinates.
        """
        # If states are already in coordinate form, return them directly
        if isinstance(state, tuple):
            return state

        # Otherwise, convert state index to coordinates
        i = state // self.dim[1]  # Calculate row index
        j = state % self.dim[1]   # Calculate column index
        return (i, j)

    def plot_traj_density(self, trajectories):
        """
        Visualizes the density of trajectories passing through each cell with the maximum
        value represented as red and the minimum as white.

        Args:
        - trajectories: A list of trajectories, where each trajectory is a list of states.
        """
        # Initialize a canvas with zeros
        canvas = np.zeros(shape=self.dim)

        # Iterate through trajectories and increment cell counts
        for traj in trajectories:
            for sars in traj:
                state = sars[0]  # Assuming sars = (state, action, reward, next_state)
                i, j = self.idx_to_coords(state)  # Convert state index to grid coordinates
                canvas[i, j] += 1

        # Normalize the counts to [0, 1] for color mapping
        max_count = np.max(canvas)
        if max_count > 0:
            normalized_canvas = canvas / max_count
        else:
            normalized_canvas = canvas

        # Create a colormap from white (min) to red (max)
        cmap_ = colors.LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])
        
        # Plotting
        plt.figure(figsize=(5, 5))
        im = plt.imshow(normalized_canvas, interpolation='none', aspect='equal', cmap=cmap_)

        ax = plt.gca()

        # Major ticks and labels
        ax.set_xticks(np.arange(0, self.dim[0], 1))
        ax.set_yticks(np.arange(0, self.dim[1], 1))
        ax.set_xticklabels(np.arange(1, self.dim[0] + 1, 1))
        ax.set_yticklabels(np.arange(1, self.dim[1] + 1, 1))

        # Minor ticks for grid lines
        ax.set_xticks(np.arange(-.5, self.dim[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.dim[1], 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

        plt.colorbar(im)  # Show color scale
        plt.show()

    
    def plot_traj_start_end(self, traj, start_idx, end_idx):

#         canvas = np.zeros(shape=self.dim)

#         # Add start states
#         for start in self.starts:
#             canvas[tuple(start)] = +0.25

#         # Add end states
#         for coord, reward in self.terminals.items():
#             canvas[coord] = reward

#         # Add obstacles
#         for obstacle in self.obstacles:
#             canvas[tuple(obstacle)] = -0.25

#         # Mark player position
#         canvas[tuple(self.player_pos)] = 0.5
            
        cmap_ = colors.ListedColormap(['black', 'gray', 'lime', 'red', 'cyan']) 
    # 0 - bg
    # 1 - wall
    # 2 - goal
    # 3 - pit
    # 4 - agent
    
        canvas = np.zeros(shape=self.dim)

        # Add end states
        for coord, reward in self.terminals.items():
            if reward > 0:
                canvas[coord] = 2
            elif reward < 0:
                canvas[coord] = 3
        
        
        # Add obstacles
        for obstacle in self.obstacles:
            canvas[tuple(obstacle)] = 1
        
        # Add agent
        canvas[tuple(self.player_pos)] = 4
            
    
        action_render = ['<', '^', '>', 'v']

        plt.figure(figsize=(2,2))
        fig, ax = plt.subplots()
        im = ax.imshow(canvas)

        text = []
        for i in range(self.dim[0]):
            text_ = []
            for j in range(self.dim[1]):
                text_.append('')
            text.append(text_)

        for idx, sars_ in enumerate(traj):
            i, j = idx_to_coords(sars_[0], grid_dim=self.dim)
            if text[i][j] != '':
                text[i][j] = text[i][j] + '\n' + action_render[sars_[1]] + ',' + str(idx + start_idx)
                
            else:
                text[i][j] = action_render[sars_[1]] + ',' + str(idx + start_idx)
                
        for idx, sars_ in enumerate(traj):
            i, j = idx_to_coords(sars_[0], grid_dim=self.dim)
            ax.text(j, i, text[i][j],
                    ha="center", va="center", color="w", size='small')
        plt.show()

    
    def plot_policy(self, policy):

#         canvas = np.zeros(shape=self.dim)

#         # Add start states
#         for start in self.starts:
#             canvas[tuple(start)] = +0.25

#         # Add end states
#         for coord, reward in self.terminals.items():
#             canvas[coord] = reward

#         # Add obstacles
#         for obstacle in self.obstacles:
#             canvas[tuple(obstacle)] = -0.25

#         # Mark player position
#         canvas[tuple(self.player_pos)] = 0.5

        cmap_ = colors.ListedColormap(['black', 'gray', 'lime', 'red', 'cyan']) 
    # 0 - bg
    # 1 - wall
    # 2 - goal
    # 3 - pit
    # 4 - agent
    
        canvas = np.zeros(shape=self.dim)

        # Add end states
        for coord, reward in self.terminals.items():
            if reward > 0:
                canvas[coord] = 2
            elif reward < 0:
                canvas[coord] = 3
        
        
        # Add obstacles
        for obstacle in self.obstacles:
            canvas[tuple(obstacle)] = 1
        
        # Add agent
        canvas[tuple(self.player_pos)] = 4
            
            
        action_render = ['<', '^', '>', 'v']

        plt.figure(figsize=(2,2))
        fig, ax = plt.subplots()
        im = ax.imshow(canvas)

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                ax.text(j, i, str(action_render[policy[i][j]]),
                        ha="center", va="center", color="w", size='small')
        plt.show()

    def plot_traj_densities(self, trajectories_lists):
        """
        Visualizes the density of trajectories for four lists of trajectories, 
        plotting each in a subplot in a 2x2 layout. Each plot's maximum value 
        is represented as red and the minimum as white.

        Args:
        - trajectories_lists: A list containing four lists of trajectories, 
          where each trajectory is a list of states.
        """
        # Setup the 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()  # Flatten to make iteration easier

        # Create a colormap from white (min) to red (max)
        cmap_ = colors.LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

        for idx, trajectories in enumerate(trajectories_lists):
            # Initialize a canvas with zeros for each list of trajectories
            canvas = np.zeros(shape=self.dim)

            # Iterate through trajectories and increment cell counts
            for traj in trajectories:
                for sars in traj:
                    state = sars[0]  # Assuming sars = (state, action, reward, next_state)
                    i, j = self.idx_to_coords(state)  # Convert state index to grid coordinates
                    canvas[i, j] += 1

            # Normalize the counts to [0, 1] for color mapping
            max_count = np.max(canvas)
            if max_count > 0:
                normalized_canvas = canvas / max_count
            else:
                normalized_canvas = canvas

            # Plotting each subplot
            im = axs[idx].imshow(normalized_canvas, interpolation='none', aspect='equal', cmap=cmap_)
            axs[idx].set_title(f'Trajectory Group {idx+1}')

            # Major ticks and labels
            axs[idx].set_xticks(np.arange(0, self.dim[0], 1))
            axs[idx].set_yticks(np.arange(0, self.dim[1], 1))
            axs[idx].set_xticklabels(np.arange(1, self.dim[0] + 1, 1))
            axs[idx].set_yticklabels(np.arange(1, self.dim[1] + 1, 1))

            # Minor ticks for grid lines
            axs[idx].set_xticks(np.arange(-.5, self.dim[0], 1), minor=True)
            axs[idx].set_yticks(np.arange(-.5, self.dim[1], 1), minor=True)

            # Gridlines based on minor ticks
            axs[idx].grid(which='minor', color='w', linestyle='-', linewidth=2)

        # Adjust layout
        plt.tight_layout()
        # Adding a colorbar
        fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)

        plt.show()

    
# Implementation of agent
class Agent:
    """
        Agent trained using Dyna-Q to generate trajectories
    """

    def __init__(self, env):
        self.env = env
        self.Q = {}
        self.model = {}

        for i in range(env.dim[0]):
            for j in range(env.dim[1]):
                s = coords_to_idx(i, j, grid_dim=env.dim)
                self.Q[s] = []
                self.model[s] = []
                for a in range(4):
                    self.Q[s] += [np.random.random()]
                    self.model[s] += [np.random.random()]

    def train(self, episode_nums, env, alpha, gamma, eval_epochs, render=True):
        """Trains agent using Dyna-Q algorithm"""
        total_reward = 0
        episode_num = 0
        episode_len = 0
        running_average = []
        episode_lengths = []

        ActionMemory = {}
        StateMemory = []

        while episode_num < episode_nums:
            s = coords_to_idx(env.player_pos[0], env.player_pos[1], grid_dim=env.dim)
            a = self.sample_action(s)
            p_s = s
            StateMemory.append(s)
            if s not in ActionMemory:
                ActionMemory[s] = []
            ActionMemory[s] += [a]

            s, r, done, _ = env.step(a)
            s = coords_to_idx(s[0], s[1], grid_dim=env.dim)
            episode_len += 1

            total_reward += r
            self.Q[p_s][a] += alpha * (r + (gamma * np.max(self.Q[s])) - self.Q[p_s][a])
            self.model[p_s][a] = (r, s)
            if done:
                env.reset()
                episode_num += 1
                episode_lengths.append(episode_len)
                episode_len = 0

                running_average.append(total_reward)
                total_reward = 0
            for n in range(eval_epochs):
                s1 = np.random.choice(StateMemory)
                a1 = np.random.choice(ActionMemory[s1])
                r1, s_p1 = self.model[s1][a1]
                self.Q[s1][a1] += alpha * (r1 + (gamma * np.max(self.Q[s_p1])) - self.Q[s1][a1])
        logging.info('Finished training the agent')
        return running_average, episode_lengths

    def sample_action(self, s, epsilon=0.1):
        """Sampling action with epsilon"""
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[s])

    def perform(self, env, max_traj_len):  # Outputs learned behaviour
        """Generate trajectories using whatever agent has learnt"""
        s = env.reset()
        s = coords_to_idx(row=s[0], col=s[1], grid_dim=env.dim)
        done = False
        traj = []
        traj_len = 0
        while not done and traj_len < max_traj_len:
            a = self.sample_action(s)
            s_, r, done, _ = env.step(a)
            # Record the state and action
            traj.append((s, a, r, coords_to_idx(row=s_[0], col=s_[1], grid_dim=env.dim)))

            s = s_
            s = coords_to_idx(row=s[0], col=s[1], grid_dim=env.dim)
            traj_len += 1

        # Final transition: (terminal_state, any action, zero reward, terminal_state)
        traj.append((s, np.random.randint(4), 0, s))

        return traj, r


if __name__ == '__main__':
    env = Environment()
    env.reset()
    # for _ in range(100):
    #     action = np.random.choice(env.action_space)
    #     env.step(action)
    #     env.render()

    # action = 0
    # while action != -1:
    #     action = int(input("Input an action 0, 1, 2, 3"))
    #     env.step(action)
    #     env.render()

    agent = Agent(env=env)
    running_average, episode_lengths = agent.train(episode_nums=50, env=env, alpha=0.1, gamma=0.01, eval_epochs=5,
                                                   render=True)
