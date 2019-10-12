from mdp import MDP


def get_states(L, W):
	S = []
	for x in range(L):
	    for y in range(W):
	        for h in range(12):
	            S.append((x, y, h))
	return S


def get_actions():
	A = [(0, 0)]
	for move in [1, -1]:
	    for rotation in [0, -1, 1]:
	        A.append((move, rotation))
    return A


def P_function(s, a, s_):
	pass


def R_function(s, a, s_):
	pass


gamma = 0.9

L = 8
W = 8
states = get_states(L, W)
actions = get_actions()

mdp = MDP(states, actions, P_function, R_function, gamma)