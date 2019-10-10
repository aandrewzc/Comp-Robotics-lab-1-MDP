
class MDP:
	"""
	Class for a general MDP

	Attributes:
		states (list)
		actions (list)
		P_function (function(s,a,s_))
		R_function (function(s,a,s_))
		gamma (float)

	"""

	def __init__(self, states, actions, P_function, R_function, gamma):

		self.states = states
		self.actions = actions
		self.P_function = P_function
		self.R_function = R_function
		self.gamma = gamma


	def evaluate_policy(self, pi):
		"""
		Args:
			pi (dict)

		Returns:
			Q function in the form of a dict mapping each (s,a) to Q(s,a)
		"""
		pass


	def policy_iteration(self):
		"""
		Returns:
			pi_star (dict)
		"""
		pass


	def value_iteration(self):
		"""
		Returns:
			pi_star (dict)
		"""
		pass