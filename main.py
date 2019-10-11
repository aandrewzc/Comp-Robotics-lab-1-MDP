from mdp import MDP


def get_states():
	pass


def get_actions():
	pass


def P_function(s, a, s_):
	pass


def R_function(s, a, s_):
	pass


gamma = 0.9


states = get_states()
actions = get_actions()

mdp = MDP(states, actions, P_function, R_function, gamma)