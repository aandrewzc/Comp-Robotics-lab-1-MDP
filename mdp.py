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

    # note that we consider a general reward function despite question requiring reward to only depend on s'

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

        # Reward matrix induced by pi, R(i,j) = R_function(s_i, pi[s_i], s_j)
        R_matrix = np.zeros((self.NS, self.NS))

        # Transition probability matrix induced by pi, P(i,j) = P_function(s_i, pi[s_i], s_j)
        P_matrix = np.zeros((self.NS, self.NS))

        for i, s in enumerate(self.states):
        
            a = pi[s]
        
            for i_, s_ in enumerate(self.states):
        
                R_matrix[i,i_] = self.R_function(s, a, s_)
                P_matrix[i,i_] = self.P_function(s, a, s_)

        # eye(NS) corresponds to V coefficients in LHS of Bellman's equations
        # and gamma*P corresponds to V coefficients in RHS
        A = np.eye(self.NS)-self.gamma*P_matrix

        # constant terms in Bellman's equations; sum aong 1st axis corresponds to sum over s' in Bellman's equations
        # * is elementwise product and not matrix multiplication; same as np.diag(P_matrix.mul(R_matrix.T))
        b = (P_matrix*R_matrix).sum(axis=1)

        # np.linalg.solve solves Bellman's equations to give pi's value function
        V = dict(zip(self.states, np.linalg.solve(A,b)))

        # action-value function
        Q = {}

        for s in self.states:

            for a in self.actions:
                
                qval = 0
                
                for s_ in self.states:

                    # Bellman's equations denoting Q-V relation
                    qval += self.P_function(s, a, s_)*(self.R_function(s, a, s_) + self.gamma*V[s_])

                Q[(s,a)] = qval

        return V, Q


    def create_policy(self, V):
        """
        Args:
            V (dict)

        Returns:
            pi in the form of a dict mapping states to actions
        """

        # action-value function
        Q = {}

        for s in self.states:

            for a in self.actions:
                
                qval = 0
                
                for s_ in self.states:

                    # Bellman's equations denoting Q-V relation
                    qval += self.P_function(s, a, s_)*(self.R_function(s, a, s_) + self.gamma*V[s_])

                Q[(s,a)] = qval

        # policy that will be returned
        pi = {}

        # for each state s, find action a such that Q(s,a) is closest to V(s) 
        for s in self.states:

            pi[s] = self.actions[0]

            for a in self.actions:

                if np.abs(Q[(s,a)]-V[s]) < np.abs(Q[(s,pi[s])]-V[s]):

                    pi[s] = a

        return pi


    def policy_iteration(self, pi_0=None):
        """
        Args:
            pi_0 (dict): initial policy

        Returns:
            pi_star (dict)
        """

        # if starting policy not given, choose first action for each state for initial policy (arbitrary choice)
        if pi_0 is None:
            pi = {s:self.actions[0] for s in self.states}
    
        # to store optimal value function
        V_star = {}

        while True:

            # to show progress
            print(".", end="")

            # evaluate policy to get Q,V
            V, Q = self.evaluate_policy(pi)
            
            # next policy will be pi_
            pi_ = pi.copy()

            for s in self.states:
                
                # at each state s, select action a with the highest value of Q(s,a)
                for a in self.actions:

                    if Q[(s,a)] > Q[(s,pi_[s])]:

                        pi_[s] = a

            # stop when policy does not change
            if pi==pi_:

                return V, pi

            # repeat the steps with the new policy if it is different
            pi = pi_


    def value_iteration(self):
        """
        Returns:
            pi_star (dict)
        """

        # thresholf for convergence
        eps = 1e-6

        # NSxNAxNS 3d array, R_3darray[i,j,k] = R_function(s_i,a_j,s_k)
        R_3darray = np.zeros((self.NS,self.NA,self.NS))
        
        # NSxNAxNS 3d array, P_3darray[i,j,k] = P_function(s_i,a_j,s_k)
        P_3darray = np.zeros((self.NS,self.NA,self.NS))

        for si, s in enumerate(self.states):
            
            for ai, a in enumerate(self.actions):
            
                for si_, s_ in enumerate(self.states):

                    R_3darray[si,ai,si_] = self.R_function(s, a, s_)
                    P_3darray[si,ai,si_] = self.P_function(s, a, s_)

        # initial estimate of optimal value (H=0)
        V = np.zeros(self.NS)

        while True:

            # Bellman optimality operator / Bellman backup
            # R and P are stored as 3d-array with {0,1,2} as axes
            # sum along 2nd axis corresponds to sum over s' in Bellmans's equations; returns NSxNA matrix
            # matrix dot product also corresponds to sum over s'; returns NSxNA matrix
            # max along 1st axis corresponds to sum over a  in Bellmans's equations; returns NS length vector
            V_ = ((P_3darray*R_3darray).sum(axis=2) + self.gamma*(P_3darray.dot(V))).max(axis=1)

            # stop when V does not change much
            if np.abs(V_-V).max() < eps:
                break

            V = V_

        # V onverges to the optimal value function
        V_star = dict(zip(self.states, V))
        
        # getting pi_star (as an array) from v_star using argmax instead of max
        pi_star = ((P_3darray*R_3darray).sum(axis=2) + self.gamma*(P_3darray.dot(V))).argmax(axis=1)
        # converting pi_star to dict
        pi_star = {self.states[i]:self.actions[pi_star[i]] for i in range(self.NS)}

        return V_star, pi_star




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

