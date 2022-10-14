import sys

def minimax(node, depth, isMaxP, alpha, beta):
	if len(node.get_children()) == 0:
		return heuristic(node)
	
	if isMaxP:
		best = sys.maxint
		for child in node.get_children():
			current = minimax(child, depth+1, False, alpha, beta)
			best = max(current, best)
			alpha = max(alpha, best)
			if alpha > beta:
				break
		return best

	else:
		best = -sys.maxint - 1
		for child in node.get_children():
			current = minimax(child, depth+1, True, alpha, beta)
			best = min(current, best)
			beta = min(best, beta)
			if alpha > beta:
				break
		return best

def heuristic():
	pass