import hashlib
import json
import PIL
from PIL import Image
import scipy.misc as smp
import gym
from gym import wrappers
import numpy as np
import cv2
import os.path
env = gym.make('Assault-v0')
env = wrappers.Monitor(env, 'tmp/Assault-v0', force=True)

class State:
    def __init__(self, observation, centres_of_ships_prev):
        #init observation
        self.observation_orig = observation
        # img = smp.toimage(self.observation_orig)
        img = PIL.Image.fromarray(self.observation_orig)
        img.save('tmp/t.jpg')
        
        self.observation = cv2.imread('tmp/t.jpg', 0)

        #init default variable
        self.player_coords = (-100, -100)
        self.centres_of_ships      = np.empty((0, 2), int)
        self.centres_of_ships_prev = np.empty((0, 2), int)
        self.count_of_killed_ships = 0
        self.dispersion_of_ships = 0
        self.middle_position_of_ships = (-100, -100)

        #Flags
        self.is_compute_player_coords = False
        self.is_compute_player_coords         = False
        self.is_compute_centres_of_ships      = False
        self.is_compute_count_of_killed_ships = False
        self.is_compute_dispersion_of_ships = False
        self.is_compute_middle_position_of_ships = False

        self.centres_of_ships_prev = centres_of_ships_prev


    def generate_hash(self):
        dict_of_features = {}
        dict_of_features['get_player_coords'] = str(self.normalize_for_hashing_get_player_coords(self.get_player_coords()))
        dict_of_features['get_middle_position_of_ships'] = str(self.normalize_for_hashing_get_middle_position_of_ships(self.get_middle_position_of_ships()))
        dict_of_features['get_dispersion_of_ships'] = str(self.normalize_for_hashing_get_dispersion_of_ships(self.get_dispersion_of_ships()))

        return hashlib.sha1(str(json.dumps(dict_of_features, sort_keys=True)).encode('utf-8')).hexdigest()

    def normalize_for_hashing_get_player_coords(self, value_of_feature):
        bins = np.linspace(0, 160, 10)
        return np.digitize(value_of_feature[1], bins)

    def normalize_for_hashing_get_dispersion_of_ships(self, value_of_feature):
        if np.isnan(value_of_feature) or value_of_feature == 0:
            value_of_feature = -1
        bins = np.linspace(0, 50, 10)
        return np.digitize(value_of_feature, bins)

    def normalize_for_hashing_get_middle_position_of_ships(self, value_of_feature):
        bins = np.linspace(0, 160, 10)
        return np.digitize(value_of_feature[1], bins)


    def get_player_coords(self):

        if self.is_compute_player_coords:
            return self.player_coords

        img = self.observation[200:224:, :]
        
        # _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        #cv2.imshow('image', img)
        for i in range(len(contours)):
            # reomve empty area
            if cv2.contourArea(contours[i]) < 50:
                continue


            moments = cv2.moments(contours[i])
            row = np.array([[int(moments['m01'] / moments['m00']), int(moments['m10'] / moments['m00']), ]])
            self.player_coords = (200 + row[0][0], row[0][1])
            #self.centres_of_ships = np.append(self.centres_of_ships, row, axis=0)
            #cv2.circle(img, self.player_coords, 1, 255, -1)
            #cv2.imshow('image', img)
            #cv2.imwrite('tmp/output.png', img)

        self.is_compute_player_coords = True
        return self.player_coords


    def get_ships_centres(self):

        if self.is_compute_centres_of_ships:
            return self.centres_of_ships

        img = self.observation[80:200:, :]

        contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)


        for i in range(len(contours)):
            # reomve empty area
            if cv2.contourArea(contours[i]) < 100:
                continue
            # remove shot
            _, _, w, _ = cv2.boundingRect(contours[i])
            if (w < 10):
                continue

            moments = cv2.moments(contours[i])
            row = np.array([[int(moments['m01'] / moments['m00']), int(moments['m10'] / moments['m00']), ]])
            self.centres_of_ships = np.append(self.centres_of_ships, row, axis=0)
            # cv2.circle(img_raw, (row[0][0], row[0][1] + 80), 3, (255, 0, 0), -1)

        self.centres_of_ships = self.centres_of_ships + [80, 0]
        # cv2.imwrite('tmp/output%d.png'%(count), img_raw)

        self.is_compute_centres_of_ships = True
        return self.centres_of_ships

    def get_middle_position_of_ships(self):
        if self.is_compute_middle_position_of_ships:
            return self.middle_position_of_ships

        ships_centres = self.get_ships_centres()
        if ships_centres.size != 0 :
            result = np.mean(self.get_ships_centres(), axis=0).astype(int)
            self.middle_position_of_ships = (result[0], result[1])
            self.is_compute_middle_position_of_ships = True
        return self.middle_position_of_ships

    def get_dispersion_of_ships(self):
        if self.is_compute_dispersion_of_ships:
            return self.dispersion_of_ships

        if self.get_ships_centres().shape[0] > 1:
            self.dispersion_of_ships = np.std(self.get_ships_centres())
        self.is_compute_dispersion_of_ships = True
        return self.dispersion_of_ships

    def get_count_of_killed_ships(self):
        if self.is_compute_count_of_killed_ships:
            return self.count_of_killed_ships

        current_count = self.centres_of_ships.shape[0]
        prev_count    = self.centres_of_ships_prev.shape[0]
        if current_count < prev_count:
            self.count_of_killed_ships = prev_count - current_count

        self.is_compute_count_of_killed_ships = True
        return  self.count_of_killed_ships

class Reward:
    def __init__(self, curstate: State, prevstate: State, action: int, reward_from_env):
        self.curstate = curstate
        self.prevstate = prevstate
        self.action = action
        self.reward_from_env = reward_from_env

        self.reward_from_action = 0
        self.reward_from_killed_ships = 0

    def compute(self):
        #if shoot
        if self.action == 2:
            self.reward_from_action = -0.1
        # if self.curstate.get_count_of_killed_ships() > 0:
        #     self.reward_from_killed_ships = 1

        return self.reward_from_env + self.reward_from_action #+ self.reward_from_killed_ships
        #return self.reward_from_action + self.reward_from_killed_ships
#
class Q:
    def __init__(self, path_to_file, env, path_to_file_with_statistic):
        self.z = 0.7
        self.gama = 0.95

        self.path_to_file = path_to_file
        self.path_to_file_with_statistic = path_to_file_with_statistic
        self.q = {}
        if os.path.isfile(path_to_file):
            self.q = np.load(self.path_to_file, allow_pickle=True).item()
        self.env = env
        self.count_explore = 0
        self.count_sence_action = 0
        self.count_new_random_action = 0
        self.common_reward = 0
        self.common_killed_ships = 0
        self.count_shoots = 0

    def save(self):
        np.save(self.path_to_file, self.q)
        if os.path.isfile(self.path_to_file_with_statistic):
            all_statistic = np.load(self.path_to_file_with_statistic)
        else:
            all_statistic = np.empty((0, 6), float)


        row = np.array([[self.count_explore, self.count_sence_action, self.count_new_random_action,
                         self.common_reward, self.common_killed_ships, self.count_shoots]])
        row = row/(self.count_explore + self.count_sence_action + self.count_new_random_action)
        all_statistic = np.append(all_statistic, row, axis=0)
        np.save(self.path_to_file_with_statistic, all_statistic)
        self.count_explore = 0
        self.count_sence_action = 0
        self.count_new_random_action = 0
        self.common_reward = 0
        self.common_killed_ships = 0
        self.count_shoots = 0

    def get_random_action_by_state(self):
        #return self.env.action_space.sample()
        return np.random.randint(1, 5)

    def get_max_action_by_state(self, state: State):
        hash  = state.generate_hash()
        if hash in self.q:
            actions = self.q[hash]
            return max(actions, key=lambda k: actions[k])
        return None

    def policy(self, state: State):
        action = self.policy_raw(state)
        if action == 2:
            self.count_shoots = self.count_shoots + 1
        return action

    def policy_raw(self, state: State):
        action = self.get_max_action_by_state(state)
        if action != None:
            if self.make_exploration():
                self.count_explore = self.count_explore + 1
                return self.get_random_action_by_state()
            self.count_sence_action = self.count_sence_action + 1
            return action

        self.count_new_random_action = self.count_new_random_action + 1
        return self.get_random_action_by_state()
    def make_exploration(self):
        # 10% exploration
        return np.random.randint(1, 100) > 90

    def remember_reward(self, reward: Reward, state: State, action: int, next_state: State):
        state_hash = state.generate_hash()
        next_state_hash = next_state.generate_hash()

        if state_hash not in self.q:
            self.q[state_hash] = {}

        cur_q = 0
        if action in self.q[state_hash]:
            cur_q = self.q[state_hash][action]
        cur_reward = reward.compute()
        self.common_reward = self.common_reward + cur_reward

        next_max_q = 0
        if next_state_hash in self.q and bool(self.q[next_state_hash]):
            next_max_q = max(self.q[next_state_hash], key=lambda k: self.q[next_state_hash][k])
            next_max_q = self.q[next_state_hash][next_max_q]

        self.common_killed_ships = self.common_killed_ships + next_state.get_count_of_killed_ships()
        #Q(S, a) = z* Q(S, a) + (1-z) * (R + gama * max(Q(S1,ai)) )
        self.q[state.generate_hash()][action] = self.z * cur_q + (1 - self.z) * (cur_reward + self.gama * next_max_q)

class Agent:
    def __init__(self, q):
        self.q   = q

    def policy(self, state: State):
        return self.q.policy(state)

    def remember_reward(self, reward: Reward, state: State, action: int, next_state: State):
        self.q.remember_reward(reward, state, action, next_state)

    def save(self):
        self.q.save()

class Debug:
    def __init__(self, state: State):
        self.state = state
        self.observation = self.state.observation_orig


    def print_debbug_with_image(self):
        print('Player centre color: RED(255, 0,0)')
        print('Ships centrs color: GREEN(0, 255, 0)')
        print('Centre of all ships: BLUE(0,0, 255)')
        self.observation[self.state.get_player_coords()] = [255, 0, 0]
        for center in self.state.get_ships_centres():
            self.observation[center[0], center[1]] = [0, 255, 0]

        middle_position = self.state.get_middle_position_of_ships()
        self.observation[middle_position[0], middle_position[1]] = [0, 0, 255]
        img = smp.toimage(self.observation)
        img.show()


    def print_debbug(self):
        print('Player coords: ', self.state.get_player_coords())
        print('Ships coords: ', self.state.get_ships_centres())
        print('Centre of all ships coords: ', self.state.get_middle_position_of_ships())
        print('Std of all ships: ', self.state.get_dispersion_of_ships())
        print('Count of killed ships: ', self.state.get_count_of_killed_ships())


q = Q('q.npy',env, 'stats.npy')
agent = Agent(q=q)
all_dispersion_of_ships = np.empty((0, 1), float)

for i_episode in range(20):
    print("Episode#", i_episode)
    observation = env.reset()
    # init state
    state = State(observation, np.empty((0, 2), int))
    for t in range(1000000):
        env.render()
        # generate action
        action = agent.policy(state)

        observation, reward_from_env, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

        # init new state
        prev_state = state
        state = State(observation, prev_state.get_ships_centres())
        debug = Debug(state)
        # debug.print_debbug_with_image()

        # compute reward
        reward = Reward(state, prev_state, action, reward_from_env)

        # update q
        agent.remember_reward(reward, prev_state, action, state)
    agent.save()