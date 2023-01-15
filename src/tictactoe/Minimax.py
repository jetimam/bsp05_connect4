from copy import deepcopy
import numpy as np

class Minimax():
	def __init__(self):
		pass
	def search(self, obs, qtable, node, depth, isMaxP, alpha, beta):
		if len(self.get_children(node, depth)) == 0 or depth == 2:
			return self.heuristic(qtable, node, obs, depth)
		if isMaxP:
			best = [0, -1] # (action, value)
			children = self.get_children(node, depth)
			counter = 0
			for child in children: # (state, action to get to the state)
				current = self.search(obs, qtable, child[0], depth+1, False, alpha, beta) # value
				if current is None: continue
				if current[1] > best[1]:
					best[0] = counter
					best[1] = current[1]
				if best[1] >= beta: break
				if best[1] > alpha: alpha = best[1]
				counter += 1
			# print('depth: ' + str(depth) + 'best: ' + str(best))
			return best
		else:
			best = [0, 1] # (action, value)
			children = self.get_children(node, depth)
			counter = 0
			for child in children:
				current = self.search(obs, qtable, child[0], depth+1, True, alpha, beta)
				if current is None: continue
				if current[1] < best[1]:
					best[0] = counter
					best[1] = current[1]
				if best[1] <= alpha: break
				if best[1] < beta: beta = best[1]
				counter += 1
			# print('depth: ' + str(depth) + 'best: ' + str(best))
			return best

	def get_children(self, node, depth):
		depth = depth+1
		pid = 1
		if depth % 2 == 1: pid = 2
		node_p = list(map(list, node))
		children = []
		counter = 0
		for i in range(len(node_p)):
			for j in range(len(node_p)):
				if node_p[i][j] == 0:
					new_child = deepcopy(node_p)
					new_child[i][j] = (pid)
					new_child = tuple(map(tuple, new_child))
					children.append((new_child, counter))
					counter += 1
		return children # list(tuple(state, action))

	def sort_children():
		pass

	def heuristic(self, qtable, node, obs, depth): # h gets passed up, but not action. action needs to be relative action, not absolute action
		if node not in qtable:
			return None
		indices = []
		counter = 0
		for i in range(3):
			for j in range(3):
				if node[i][j] == 0:
					indices.append(counter)
				counter += 1
		p = [(qtable[node][i], i) for i in range(9) if i in indices]
		p.sort()
		h = p[-1][0]
		if h == 0.0:
			return None
		if depth%2 == 1: h = -h
		a = p[-1][1]
		return [a, h]