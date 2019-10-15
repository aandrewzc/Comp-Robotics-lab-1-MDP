from mdp import MDP
import matplotlib.pyplot as plt

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


def move(s, a, L, W):
	"""
	Perform an action, without the pre-rotation error
	Returns s_
	"""

	# check heading and increment position by +/- 1
	if 2 <= s[2] <= 4:
	    x = s[0] + a[0]
	    y = s[1]

	elif 5 <= s[2] <= 7:
	    x = s[0]
	    y = s[1] - a[0]

	elif 8 <= s[2] <= 10:
	    x = s[0] - a[0]
	    y = s[1]

	else: # 11, 0, 1
	    x = s[0] 
	    y = s[1] + a[0]

	# boundary checks (can't fall off the grid)
	if x < 0 or x >= L:
	    x = s[0]
	if y < 0 or y >= W:
	    y= s[1]

	# set the new heading (increment by +/- 1)
	h = (s[2] + a[1]) % 12

	s_ = (x, y, h)
	return s_


def p_sa(p_error, s, a, s_):
	"""
	Returns likeliihood of state s_ given the error prob, current state, and an action 
	Errors can happen if the robot pre-rotates left or right
	This function computes the next state in each case and compares it to s_

	"""

	if a[0] == 0:
		# If no movement occurs, the current state must equal the next state
	    if s == s_:
	        prob = 1
	    else:
	        prob = 0

	# Errors may happen if movement occurs
	else:
	    # New initial states with pre-rotation error
	    s_left = (s[0], s[1], s[2]-1) 
	    s_right = (s[0], s[1], s[2]+1)

	    # Compute all possible next states
	    s_next = move(s, a, L, W)
	    s_next_left = move(s_left, a, L, W)
	    s_next_right = move(s_right, a, L, W)

	    # Compare s_ with next states and increment probability
	    prob = 0
	    if s_ == s_next:
	        prob += (1 - 2*p_error)
	    if s_ == s_next_left:
	        prob += p_error
	    if s_ == s_next_right:
	        prob += p_error

	return prob


def next_state(p_error, s, a):
	# Find all the next possible states, then determine which has highest probability
	s_next = [
		(s[0], s[1], s[2]),
		(s[0], s[1], s[2]+a[1]),
		(s[0]+a[0], s[1], s[2]),
		(s[0]+a[0], s[1], s[2]+a[1]),
		(s[0], s[1]+a[0], s[2]),
		(s[0], s[1]+a[0], s[2]+a[1]),
	]

	max_p = 0
	for state in s_next:
		p = p_sa(p_error, s, a, state)
		if p > max_p:
			max_p = p
			s_ = state

	return s_


gamma = 0.9

L = 8
W = 8

# Testing p_sa and next_state functions here
prob = p_sa(0, (0,0,0), (1,1), (0,1,1))
print(prob)
s = next_state(0.025, (7,7,11), (-1,0))
print(s)

states = get_states(L, W)
actions = get_actions()

mdp = MDP(states, actions, P_function, R_function, gamma)


##########################
#reward function R(s)

def reward(s_x, s_y):
    #out of the world
    if ( s_x >= L or s_y >= W): 
        raise ValueError("Exceed World Size")
        
    #win!
    if (s_x == 5 or s_y == 6):
        return 1
    
    #boundary 
    if (s_x == 0 or s_x == W-1 or s_y == 0 or s_y == L-1):
        return -100
    
    #lane marker
    if ((s_x == 3 and s_y == 4) or (s_x == 3 and s_y == 5) or (s_x == 3 and s_y == 6)): 
        return -10
    return 0


##########################
#plotter
#receives input as a list of all previous states i.e. [(x1,y1,h1), (x2,y2,h2)...]
#plots location and orientation of the robot throughout the entire time history

#example input
spot = np.array([(1,1,1),(1,2,2),(1,3,3),(2,3,5,),(2,4,10),(3,4,12),(3,5,4)])

def plotter(spot):
    #x = [1,2,3]
    spot_x_y = []
    for i in spot:
        spot_x_y += [(i[0],i[1])]

     
    spot_x_y = np.array(spot_x_y)
    
    #plot path
    plt.plot(*spot_x_y.T)
    
    #plot robot + orientation
    for i in spot:
        plt.plot(i[0], i[1], marker=(3, 0, i[2]/12*360), markersize=20, linestyle='None')

    plt.xlim([0,W-1])
    plt.ylim([0,L-1])
    plt.grid()
    plt.show()
    plt.savefig("robotMap.jpg")
    

plotter(spot)
