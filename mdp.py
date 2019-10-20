import numpy as np
import sys

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

        # NSxNAxNS 3d array, R_3darray[i,j,k] = R_function(s_i,a_j,s_k)
        self.R_3darray = np.zeros((self.NS,self.NA,self.NS))
        
        # NSxNAxNS 3d array, P_3darray[i,j,k] = P_function(s_i,a_j,s_k)
        self.P_3darray = np.zeros((self.NS,self.NA,self.NS))

        # store function values so we don't need to call them again and again
        for si, s in enumerate(states):

            print(".", end="")
            sys.stdout.flush()

            for ai, a in enumerate(actions):

                for si_, s_ in enumerate(states):

                    self.R_3darray[si,ai,si_] = self.R_function(s,a,s_)
                    self.P_3darray[si,ai,si_] = self.P_function(s,a,s_)

        print()


    def evaluate_policy_accurate(self, pi):
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

        for si, s in enumerate(self.states):
        
            ai = self.actions.index(pi[s])

            R_matrix[si,:] = self.R_3darray[si, ai, :]
            P_matrix[si,:] = self.P_3darray[si, ai, :]
        
        # eye(NS) corresponds to V coefficients in LHS of Bellman's equations
        # and gamma*P corresponds to V coefficients in RHS
        A = np.eye(self.NS)-self.gamma*P_matrix

        # constant terms in Bellman's equations; sum aong 1st axis corresponds to sum over s' in Bellman's equations
        # * is elementwise product and not matrix multiplication; same as np.diag(P_matrix.mul(R_matrix.T))
        b = (P_matrix*R_matrix).sum(axis=1)

        # np.linalg.solve solves Bellman's equations to give pi's value function
        V_arr = np.linalg.solve(A,b)
        V = dict(zip(self.states, V_arr))

        # action-value function
        Q = {}

        for si, s in enumerate(self.states):

            for ai, a in enumerate(self.actions):
                
                Q[(s,a)] = self.P_3darray[si,ai,:].dot(self.R_3darray[si,ai,:]+self.gamma*V_arr)

        return V, Q


    def evaluate_policy_fast(self, pi):
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

        for si, s in enumerate(self.states):
        
            ai = self.actions.index(pi[s])

            R_matrix[si,:] = self.R_3darray[si, ai, :]
            P_matrix[si,:] = self.P_3darray[si, ai, :]

        # expected reward vector, same for each iteration
        exp_reward = (P_matrix*R_matrix).sum(axis=1)

        # start with value estimate of zero
        V = np.zeros(self.NS)
        V_ = exp_reward + self.gamma*P_matrix.dot(V)

        # threshold for convergence
        eps = 1e-6

        # do bellman back-up untill value function converges
        while np.abs(V-V_).max() > eps:

            V = V_
            V_ = exp_reward + self.gamma*P_matrix.dot(V)

        # convert V to dict
        V = dict(zip(self.states, V))

        # action-value function
        Q = {}

        for si, s in enumerate(self.states):

            for ai, a in enumerate(self.actions):
                
                Q[(s,a)] = (self.P_3darray[si,ai,:].dot(self.R_3darray[si,ai,:])+self.gamma*V_arr)

        return V, Q


    def create_policy(self, V):
        """
        Args:
            V (dict)

        Returns:
            pi in the form of a dict mapping states to actions
        """

        # converting V to an array
        V_arr = np.array([V[s] for s in self.states])

        # action-value function
        Q = {}

        for si, s in enumerate(self.states):

            for ai, a in enumerate(self.actions):
                
                Q[(s,a)] = self.P_3darray[si,ai,:].dot(self.R_3darray[si,ai,:]+self.gamma*V_arr)

        # policy that will be returned
        pi = {}

        # for each state s, find action a such that Q(s,a) is closest to V(s) 
        for si, s in enumerate(self.states):

            pi[s] = self.actions[0]

            for a in self.actions:

                if np.abs(Q[(s,a)]-V_arr[si]) < np.abs(Q[(s,pi[s])]-V_arr[si]):

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
        else:
            pi = pi_0
    
        # to store optimal value function
        V_star = {}

        # there may be multiple optimal policies and policy iteration may oscillate between them due to FPE
        # so we need value to increase by a threshold eps at each iteration
        eps = 1e-6

        while True:

            # to show progress
            print(".", end="")
            sys.stdout.flush()

            # evaluate policy to get Q,V
            V, Q = self.evaluate_policy_accurate(pi)
            
            # next policy will be pi_
            pi_ = pi.copy()

            for s in self.states:
                
                # at each state s, select action a with the highest value of Q(s,a)
                for a in self.actions:

                    if Q[(s,a)] > Q[(s,pi_[s])] + eps:

                        pi_[s] = a

            # stop when policy does not change
            if pi==pi_:
                print()
                return V, pi

            # repeat the steps with the new policy if it is different
            pi = pi_


    def value_iteration(self):
        """
        Returns:
            pi_star (dict)
        """

        # threshold for convergence
        eps = 1e-6

        # initial estimate of optimal value (H=0)
        V = np.zeros(self.NS)

        while True:

            # shows progress
            print(".", end="")
            sys.stdout.flush()

            # Bellman optimality operator / Bellman backup
            # R and P are stored as 3d-array with {0,1,2} as axes
            # sum along 2nd axis corresponds to sum over s' in Bellmans's equations; returns NSxNA matrix
            # matrix dot product also corresponds to sum over s'; returns NSxNA matrix
            # max along 1st axis corresponds to sum over a  in Bellmans's equations; returns NS length vector
            V_ = ((self.P_3darray*self.R_3darray).sum(axis=2) + self.gamma*(self.P_3darray.dot(V))).max(axis=1)

            # stop when V does not change much
            if np.abs(V_-V).max() < eps:
                print()
                break

            V = V_

        # V onverges to the optimal value function
        V_star = dict(zip(self.states, V))
        
        # getting pi_star (as an array) from v_star using argmax instead of max
        pi_star = ((self.P_3darray*self.R_3darray).sum(axis=2) + self.gamma*(self.P_3darray.dot(V))).argmax(axis=1)
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

    # mdp.evaluate_policy_accurate({0:0, 1:0})

    print(mdp.value_iteration())

    print(mdp.policy_iteration())

