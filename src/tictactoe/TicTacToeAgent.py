import numpy as np
import random

class TicTacToeAgent:
	def __init__(self, learning_rate, epsilon, epsilon_decay, qtable={}):
		self.qtable = qtable
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay

	def get_epsilon_action(self, obs, new_state_p):
		indices = np.argwhere(obs['action_mask']==1).flatten()
		if np.max(self.qtable[new_state_p][indices]) == 0 or random.random() < self.epsilon: #exploration
			action = np.random.choice(indices)
		else: #exploitation
			p = [(self.qtable[new_state_p][i], i) for i in range(9) if i in indices]
			p.sort()
			action = p[-1][1]
		return action

	def get_best_action(self, obs, new_state_p):
		indices = np.argwhere(obs['action_mask']==1).flatten()
		p = [(self.qtable[new_state_p][i], i) for i in range(9) if i in indices]
		p.sort()
		action = p[-1][1]
		return action

	def get_random_action(self, obs):
		indices = np.argwhere(obs['action_mask']==1).flatten()
		action = random.choice(indices)
		return action

	def update(self, state_p, new_state_p, action, reward, gamma)-> None:
		if new_state_p not in self.qtable:
			self.qtable[new_state_p] = np.zeros(9)
		if state_p is not None and action is not None:
			self.qtable[state_p][action] = self.qtable[state_p][action] * (1 - self.learning_rate) + self.learning_rate * (reward + gamma * np.max(self.qtable[new_state_p]))

	def decay_epsilon(self, min_epsilon)-> None:
		self.epsilon = max(min_epsilon, self.epsilon-self.epsilon_decay)