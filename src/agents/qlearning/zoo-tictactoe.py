# import gym
import numpy as np
import random
from pettingzoo.classic import tictactoe_v3

max_episodes = 10000
max_steps = 100

learning_rate = 0.1
gamma = 0.99 #discount rate
epsilon = 0.99
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_rate = 0.001

rewards_global = []
env = tictactoe_v3.env()
state = env.reset()

print(env.action_space(env.possible_agents[0]).sample())
print(env.observation_space(env.possible_agents[0])['observation'].shape)
print(type(env.action_space(env.possible_agents[0]).n))

q_table = np.zeros((env.observation_space(env.possible_agents[0]).shape, env.action_space(env.possible_agents[0]).n))

for episode in range(max_episodes):
	state = env.reset()
	done = False
	episode_reward = 0

	for agent in env.agent_iter(max_epsilon):
		obs, reward, done, truncation, info = env.last()
		print(env.action_space)
		if random.random() > epsilon: #exploitation vs exploration balance
			action = np.argmax(q_table[state,:])
		else:
			action = env.action_space
		
		env.step(action)

		q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))
			
		state = new_state
		episode_reward += reward

		if done:
			break
	
	epsilon = min_epsilon + (max_epsilon - min_epsilon * np.exp(-epsilon_decay_rate * episode))

	rewards_global.append(episode_reward)

n = 1000
rewards_per_n_episodes = np.split(np.array(rewards_global), max_episodes/n)

for rewards in rewards_per_n_episodes:
	print(n, "-> ", str(sum(rewards/1000)), sep='')
	n += 1000

print()

print(q_table)