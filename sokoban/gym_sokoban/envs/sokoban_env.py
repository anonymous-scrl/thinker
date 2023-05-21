import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb, pre_load_imgs
import numpy as np
import random
import pkg_resources

class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=9999,
                 num_boxes=4,
                 num_gen_steps=None):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.01
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(4)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        self.imgs = pre_load_imgs()

        # Initialize Room
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action in [0, 1, 2, 3]

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_player = False
        moved_box = False
        # All push actions are in the range of [0, 3]
        if action < 4:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode='rgb_array')

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, gen_room=False, room_id=-1, second_player=False):
        if(gen_room):
            try:
                self.room_fixed, self.room_state, self.box_mapping = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    second_player=second_player
                )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                return self.reset()
            self.player_position = np.argwhere(self.room_state == 5)[0]
        else:
            resource_package = __name__

            room_id = random.randint(0, 900*1000-1) if room_id == -1 else room_id
            file_name = pkg_resources.resource_filename(resource_package, '/'.join(('boxoban-levels', 'unfiltered',
                        "train", "{0:0=3d}.txt".format(int(room_id / 1000)))))

            line_num = (room_id % 1000) * 12 + 1
            self.room_state = np.zeros((10, 10), dtype=np.int64)

            with open(file_name) as fp:
                for i, line in enumerate(fp):
                    if i >= line_num and i < line_num + 10:
                        for j in range(0, 10): self.room_state[i - line_num, j] = \
                            (SYM_LOOKUP[line[j]])
                    elif i >= line_num + 10:
                        break
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.room_fixed = np.copy(self.room_state)            
            self.room_fixed[self.player_position[0], self.player_position[1]] = 1
            self.room_fixed[self.room_fixed == 4] = 1

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = room_to_rgb(self.room_state, surfaces=self.imgs, room_structure=self.room_fixed)
        return starting_observation

    def render(self, mode='human', close=None):
        assert mode in RENDERING_MODES

        img = self.get_image(mode)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode):

        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=4)
        else:
            img = room_to_rgb(self.room_state, surfaces=self.imgs, room_structure=self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    def clone_state(self):

        return {"sokoban_player_position": np.copy(self.player_position),
               "sokoban_room_state": np.copy(self.room_state),
               "sokoban_room_fixed": np.copy(self.room_fixed),
               "sokoban_num_env_steps": self.num_env_steps,
               "sokoban_reward_last": self.reward_last,
               "sokoban_boxes_on_target": self.boxes_on_target}  

    def restore_state(self, state):
        self.player_position = np.copy(state["sokoban_player_position"])
        self.room_state = np.copy(state["sokoban_room_state"])
        self.room_fixed = np.copy(state["sokoban_room_fixed"])
        self.num_env_steps = state["sokoban_num_env_steps"]
        self.reward_last = state["sokoban_reward_last"]
        self.boxes_on_target = state["sokoban_boxes_on_target"]

ACTION_LOOKUP = [
    'PUSH UP',
    'PUSH DOWN',
    'PUSH LEFT',
    'PUSH RIGHT']

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human']

SYM_LOOKUP = {"#":0, #wall 
              " ":1, #empty space
              ".":2, #box target
              "$":4, #box not on target
              "@":5} #player