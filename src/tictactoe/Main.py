import json
import sys
import numpy as np
from pettingzoo.classic import tictactoe_v3
from TicTacToeAgent import TicTacToeAgent
from Minimax import minimax

def merge_states(obs):
	state_p = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	for t1 in range(len(obs)):
		for t2 in range(len(obs[0])):
			if obs[t1, t2, 0] == 1:
				state_p[t1][t2] = 1
			elif obs[t1, t2, 1] == 1:
				state_p[t1][t2] = 2

	return tuple(map(tuple, state_p))

max_episodes = 10000
max_steps = 100
learning_rate = 0.1
gamma = 0.99 #discount rate
epsilon = 0.99
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_rate = 0.0002
rewards_global_0 = []
rewards_global_1 = []
env = tictactoe_v3.env()
env.reset()
np.random.seed(57)

# minimax get_children() for tictactoe using s` DONE
# report DONE
# merge the tables (player2)
	# train player2 against pre-trained player1 -> json to dump the qtable
# interactive player DONE
# ===================
# bootstrap

p1 = TicTacToeAgent(learning_rate, epsilon, epsilon_decay_rate)
p2 = TicTacToeAgent(learning_rate, epsilon, epsilon_decay_rate)
p_id = 0

def qlearning():
	obs, reward[p_id], done[p_id], _, _ = env.last()
	episode_reward[p_id] += reward[p_id]
	new_state_p[p_id] = merge_states(env.observe('player_1')['observation'])
	p1.update(state_p[p_id], new_state_p[p_id], actions[p_id], reward[p_id], gamma)
	state_p[p_id] = new_state_p[p_id]
	if done[p_id]:
		actions[p_id] = None
	else:
		actions[p_id] = p1.get_best_action(obs, new_state_p[p_id])
	env.step(actions[p_id])
	return done[(p_id+1) % 2]

def randomized():
	obs, _, done[p_id], _, _ = env.last()
	if done[p_id]:
		actions[p_id] = None
	else:
		actions[p_id] = p2.get_random_action(obs)
	env.step(actions[p_id])
	return done[(p_id+1) % 2]

def minimaxed():
	obs, _, done[p_id], _, _ = env.last()
	mini_state_p = merge_states(env.observe('player_1')['observation'])
	actions[p_id] = minimax(obs, static_qtable, p_id, mini_state_p, 0, True, sys.maxint, -sys.maxint - 1)
	env.step(actions[p_id])
	return done[(p_id+1) % 2]

def interactive():
	env.render()
	print('Make an action: Top left is 0, bottom right is 8.')
	actions[p_id] = int(input())
	env.step(actions[p_id])
	return done[(p_id+1) % 2]

for episode in range(max_episodes):
	print('episode', episode)
	env.reset()
	state_p = [None, None]
	new_state_p = [None, None]
	actions = [None, None]
	reward = [0, 0]
	done = [False, False]
	episode_reward = [0, 0]
	round = 0
	for agent in env.agent_iter(max_steps):
		if agent == 'player_1':
			if qlearning():
				break
		if agent == 'player_2':
			if interactive():
				break
		p_id = (p_id + 1) % 2
	
	p1.decay_epsilon(min_epsilon)
	p2.decay_epsilon(min_epsilon)

	rewards_global_0.append(episode_reward[0])
	rewards_global_1.append(episode_reward[1])

qtable_merged = p1.qtable
qtable_merged.update(p2.qtable)


n = 1000
rewards_per_n_episodes_0 = np.split(np.array(rewards_global_0), max_episodes/n)
rewards_per_n_episodes_1 = np.split(np.array(rewards_global_1), max_episodes/n)
print('Player1:')
for rewards in rewards_per_n_episodes_0:
	print(n, "-> ", str(sum(rewards/1000)), sep='')
	n += 1000
n = 1000
print('Player2:')
for rewards in rewards_per_n_episodes_1:
	print(n, "-> ", str(sum(rewards/1000)), sep='')
	n += 1000

# for key in qtable_merged:
# 	qtable_merged[key] = qtable_merged[key].tolist()
	
# with open('result.json', 'w') as fp:
# 	json.dump(qtable_merged, fp)