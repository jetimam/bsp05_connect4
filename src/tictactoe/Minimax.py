import sys
from copy import deepcopy
import numpy as np

def minimax(obs, qtable, pid, node, depth, isMaxP, alpha, beta):
	def search(obs, qtable, pid, node, depth, isMaxP, alpha, beta):
		if len(get_children(node, pid)) == 0 or depth == 3:
			return heuristic(qtable, node, obs)
		if isMaxP:
			best = sys.maxsize
			for child in get_children(node, pid):
				pid = (pid + 1) % 2
				current = search(obs, qtable, pid, child, depth+1, False, alpha, beta)
				best = max(current, best)
				alpha = max(alpha, best)
				if alpha > beta:
					break
			return best
		else:
			best = -sys.maxsize - 1
			for child in get_children(node, pid):
				pid = (pid + 1) % 2
				current = search(obs, qtable, pid, child, depth+1, True, alpha, beta)
				best = min(current, best)
				beta = min(best, beta)
				if alpha > beta:
					break
			return best

	def get_children(node, pid):
		node_p = list(map(list, node))
		children = []
		for i in range(len(node_p)):
			for j in range(len(node_p)):
				if node_p[i][j] == 0:
					temp = deepcopy(node_p)
					temp[i][j] = (pid+1)
					temp = tuple(map(tuple, temp))
					children.append(temp)
		return children

	def heuristic(qtable, node, obs):
		if node in qtable:
			indices = np.argwhere(obs['action_mask']==1).flatten()
			p = [(qtable[node][i], i) for i in range(9) if i in indices]
			p.sort()
			action = p[-1][1]
			return qtable[node][action]
		else: #default value in case node hasnt been explored yet.
			return 0

	return search(obs, qtable, pid, node, depth, isMaxP, alpha, beta)