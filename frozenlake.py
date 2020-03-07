import gym

total_episodes = 5000
total_test_episodes = 100
max_steps = 99 
alpha= 0.7 # Learning rate 
gamma = 0.8 # Discounting rate 
epsilon = 1.0 # Exploration rate
decay_rate = 0.01 # Exponential decay rate

env = gym.make('FrozenLake-v0')
for i_episode in range(1):
    observation = env.reset()
    totalReward = 0
    step = 0
    for step in range(max_steps):
        env.render()
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # if reward != 0:
        #     print(reward)
        totalReward += reward
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            print(totalReward)
            break
env.close()