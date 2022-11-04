import gym
import numpy as np
import random

max_episodes = 10000
max_steps = 100

learning_rate = 0.1
gamma = 0.99 # discount rate
epsilon = 0.99
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_rate = 0.001

rewards_global = []
env = gym.make('CliffWalking-v0')
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(max_episodes):
	state, _ = env.reset()
	done = False
	episode_reward = 0

	for step in range(max_steps):
		if random.random() > epsilon: #exploitation vs exploration balance
			action = np.argmax(q_table[state])
		else:
			action = env.action_space.sample()
		
		new_state, reward, done, _, info = env.step(action)

		q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))
		
		# print(epsilon)
		state = new_state
		episode_reward += reward

		if done:
			break
	
	epsilon = max(min_epsilon, epsilon-epsilon_decay_rate)

	rewards_global.append(episode_reward)

n = 1000
rewards_per_n_episodes = np.split(np.array(rewards_global), max_episodes/n)

for rewards in rewards_per_n_episodes:
	print(n, "-> ", str(sum(rewards/1000)), sep='')
	n += 1000

print()

# print(q_table)