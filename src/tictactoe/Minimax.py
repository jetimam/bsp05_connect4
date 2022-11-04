import sys
import numpy as np

def minimax(obs, qtable, pid, node, depth, isMaxP, alpha, beta):
	def search(obs, qtable, pid, node, depth, isMaxP, alpha, beta):
		if len(get_children(list(node), pid)) == 0 or depth == 3:
			return heuristic(qtable, node, obs)
		if isMaxP:
			best = sys.maxint
			for child in get_children(list(node), pid):
				current = search(obs, qtable, pid, child, depth+1, False, alpha, beta)
				best = max(current, best)
				alpha = max(alpha, best)
				if alpha > beta:
					break
			return best
		else:
			best = -sys.maxint - 1
			for child in get_children(list(node), pid):
				current = search(obs, qtable, pid, child, depth+1, True, alpha, beta)
				best = min(current, best)
				beta = min(best, beta)
				if alpha > beta:
					break
			return best

	def get_children(node, pid):
		children = []
		for i in range(len(node)):
			for j in range(len(node)):
				if node[i][j] == 0:
					temp = node
					temp[i][j] = (pid+1)
					children.append(temp)
		return children

	def heuristic(qtable, node, obs):
		indices = np.argwhere(obs['action_mask']==1).flatten()
		p = [(qtable[node][i], i) for i in range(9) if i in indices]
		p.sort()
		action = p[-1][1]
		return qtable[node][action]

	return search(obs, qtable, pid, node, depth, isMaxP, alpha, beta)