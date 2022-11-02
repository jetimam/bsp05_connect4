# import gym
import time
import csv
import json
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
epsilon_decay_rate = 0.0002

np.random.seed(57)

rewards_global = []
env = tictactoe_v3.env()
env.reset()

q_table = {}

def merge_states(obs):
	state_p = np.zeros((3, 3))
	for t1 in range(len(obs)):
		for t2 in range(len(obs[0])):
			if obs[t1, t2, 0] == 1:
				state_p[t1, t2] = 1
			elif obs[t1, t2, 1] == 1:
				state_p[t1, t2] = 2

	return str(state_p)

def learn(env, state_p, q_table, action, episode_reward):
	obs, reward, done, _, _ = env.last()
	episode_reward += reward
	new_state_p = merge_states(obs['observation'])

	if new_state_p not in q_table:
		q_table[new_state_p] = np.zeros(9)
	if state_p is not None:
		q_table[state_p][action] = q_table[state_p][action] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state_p]))				
	state_p = new_state_p

	if done:
		action = None
	else:
		indices = np.argwhere(obs['action_mask']==1).flatten()
		if np.argmax(q_table[new_state_p][indices]) == 0 or random.random() < epsilon: #exploration
			action = np.random.choice(indices)
		else: #exploitation
			p = [(q_table[new_state_p][i], i) for i in range(9) if i in indices]
			p.sort()
			action = p[-1][1]

	env.step(action)

	return [state_p, q_table, action, episode_reward]

def action_random(env):
	obs, _, done, _, _ = env.last()

	if done:
		action = None
	else:
		indices = np.argwhere(obs['action_mask']==1).flatten()
		action = np.random.choice(indices)

	env.step(action)

# minimax get_children() for tictactoe using s`
# report
# merge the tables (player2)
	# train player2 against pre-trained player1 -> json to dump the qtable
# interactive player
# ===================
# bootstrap

for episode in range(max_episodes):
	print('episode', episode)
	env.reset()
	state_p = None
	actions = [None, None]
	done = False
	episode_reward = 0

	for agent in env.agent_iter(max_steps):
		if agent == 'player_1':
			obs, reward, done, _, _ = env.last()
			episode_reward += reward
			new_state_p = merge_states(env.observe('player_1')['observation'])

			if new_state_p not in q_table:
				q_table[new_state_p] = np.zeros(9)
			if state_p is not None:
				q_table[state_p][action1] = q_table[state_p][action1] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state_p]))				
			state_p = new_state_p

			if done:
				action1 = None
			else:
				indices = np.argwhere(obs['action_mask']==1).flatten()
				if np.argmax(q_table[new_state_p][indices]) == 0 or random.random() < epsilon: #exploration
					action1 = np.random.choice(indices)
				else: #exploitation
					p = [(q_table[new_state_p][i], i) for i in range(9) if i in indices]
					p.sort()
					action1 = p[-1][1]

			env.step(action1)
		if done:
			break
		if agent == 'player_2':
			obs, _, done, _, _ = env.last()

			if done:
				action2 = None
			else:
				indices = np.argwhere(obs['action_mask']==1).flatten()
				action2 = np.random.choice(indices)

			env.step(action2)

	epsilon = max(min_epsilon, epsilon-epsilon_decay_rate)

	rewards_global.append(episode_reward)

n = 1000
rewards_per_n_episodes = np.split(np.array(rewards_global), max_episodes/n)

for rewards in rewards_per_n_episodes:
	print(n, "-> ", str(sum(rewards/1000)), sep='')
	n += 1000

# with open("myfile.txt", 'w') as f: 
#     for key, value in q_table.items(): 
#         f.write('%s:%s\n' % (key, value))

def dump(q_table):
	for key in q_table:
		q_table[key] = q_table[key].tolist()
		
	with open('result.json', 'w') as fp:
		json.dump(q_table, fp)