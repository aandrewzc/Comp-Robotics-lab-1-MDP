import numpy as np


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

		self.NS = len(states)
		self.NA = len(actions)


	def evaluate_policy(self, pi):
		"""
		Args:
			pi (dict)

		Returns:
			V function in the form of a dict mapping each s to V(s),
			Q function in the form of a dict mapping each (s,a) to Q(s,a)
		"""
		R_matrix = np.zeros((self.NS, self.NS))
		P_matrix = np.zeros((self.NS, self.NS))

		for i, s in enumerate(self.states):
		
			a = pi[s]
		
			for i_, s_ in enumerate(self.states):
		
				R_matrix[i,i_] = self.R_function(s, a, s_)
				P_matrix[i,i_] = self.P_function(s, a, s_)

		A = np.eye(self.NS)-self.gamma*P_matrix
		b = (P_matrix*R_matrix).sum(axis=1)

		V = dict(zip(self.states, np.linalg.solve(A,b)))

		Q = {}

		for s in self.states:

			for a in self.actions:
				
				qval = 0
				
				for s_ in self.actions:

					qval += self.P_function(s, a, s_)*(self.R_function(s, a, s_) + self.gamma*V[s_])

				Q[(s,a)] = qval

		# print(V, Q)

		return V, Q


	def policy_iteration(self):
		"""
		Returns:
			pi_star (dict)
		"""
		pi = {s:self.actions[0] for s in self.states}
		V_star = {}

		while True:

			V, Q = self.evaluate_policy(pi)
			pi_ = pi.copy()

			for s in self.states:
				
				for a in self.actions:

					if Q[(s,a)] > Q[(s,pi_[s])]:

						pi_[s] = a

			if pi==pi_:

				return V, pi

			pi = pi_


	def value_iteration(self):
		"""
		Returns:
			pi_star (dict)
		"""
		eps = 1e-6
		R_3darray = np.zeros((self.NS,self.NA,self.NS))
		P_3darray = np.zeros((self.NS,self.NA,self.NS))

		for si, s in enumerate(self.states):
			
			for ai, a in enumerate(self.actions):
			
				for si_, s_ in enumerate(self.states):

					R_3darray[si,ai,si_] = self.R_function(s, a, s_)
					P_3darray[si,ai,si_] = self.P_function(s, a, s_)

		V = np.zeros(self.NS)

		while True:

			# print(V)

			V_ = ((P_3darray*R_3darray).sum(axis=2) + self.gamma*(P_3darray.dot(V))).max(axis=1)

			if np.abs(V_-V).max() < eps:
				break

			V = V_


		# Q = {}

		# for si, s in enumerate(self.states):

		# 	for ai, a in enumerate(self.actions):
								
		# 		Q[(s,a)] = P_3darray[si,ai,:].dot(R_3darray[si,ai,:]+self.gamma*V)

		V_star = dict(zip(self.states, V))
		
		pi_star = ((P_3darray*R_3darray).sum(axis=2) + self.gamma*(P_3darray.dot(V))).argmax(axis=1)
		pi_star = {self.states[i]:self.actions[pi_star[i]] for i in range(self.NS)}

		# print(pi_star)
		
		return V_star, pi_star





# test
if __name__ == "__main__":

	states = list(range(10))
	actions = [0, 1]

	def P_function(s, a, s_):
		n = len(states)
		return 2*s_/(n*(n-1))

	def R_function(s, a, s_):

		return s+a+s_

	gamma = 0.9

	mdp = MDP(states, actions, P_function, R_function, gamma)

	# mdp.evaluate_policy({0:0, 1:0})

	print(mdp.value_iteration())

	print(mdp.policy_iteration())

