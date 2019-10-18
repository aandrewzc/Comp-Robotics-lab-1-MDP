from mdp import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from random import choice

def get_states():
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
    """
    Returns probability of reaching state s_,  given current state, and an action 
    Errors can happen if the robot pre-rotates left or right
    """
    # if a[0] == 0:
    #     # If no movement occurs, the current state must equal the next state
    #     if s == s_:
    #         prob = 1.0
    #     else:
    #         prob = 0.0

    # # Errors may happen if movement occurs
    # else:
        # New initial states with pre-rotation error
    s_left = (s[0], s[1], s[2]-1) 
    s_right = (s[0], s[1], s[2]+1)

    # Compute all possible next states
    s_next = next_state_no_error(s, a)
    s_next_left = next_state_no_error(s_left, a)
    s_next_right = next_state_no_error(s_right, a)

    # Compare s_ with next states and increment probability
    prob = 0.0
    if s_ == s_next:
        prob += (1 - 2*p_error)
    if s_ == s_next_left:
        prob += p_error
    if s_ == s_next_right:
        prob += p_error

    return prob


def R_function(s, a, s_):
    
    s_x = s_[0]
    s_y = s_[1]

    if ( s_x >= L or s_y >= W or s_x<0 or s_y<0): 
        raise ValueError("Exceed World Size")
        
    #win!
    if (s_x == 5 and s_y == 6):
        return 1
    
    #boundary 
    if (s_x == 0 or s_x == W-1 or s_y == 0 or s_y == L-1):
        return -100
    
    #lane marker
    if ((s_x == 3 and s_y == 4) or (s_x == 3 and s_y == 5) or (s_x == 3 and s_y == 6)): 
        return -10
    return 0


def next_state_no_error(s, a):
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


# def all_next_states(s, a):
#     # Find all the next possible states, then determine which has highest probability
#     s_next = [
#         (s[0], s[1], s[2]),
#         (s[0], s[1], s[2]+a[1]),
#         (s[0]+a[0], s[1], s[2]),
#         (s[0]+a[0], s[1], s[2]+a[1]),
#         (s[0], s[1]+a[0], s[2]),
#         (s[0], s[1]+a[0], s[2]+a[1]),
#     ]

#     # max_p = 0
#     # for state in s_next:
#     #     p = P_function(s, a, state)
#     #     if p > max_p:
#     #         max_p = p
#     #         s_ = state

#     return s_next

def next_state_draw(s, a):
    # Find all the next possible states, then determine which has highest probability

    weights = []
    next_states = []

    for s_ in states:
        p = P_function(s, a, s_)
        if p > 0:
            next_states.append(s_)
            weights.append(p)

    # print(weights)
    # print(next_states)
    assert(sum(weights) == 1.0)

    si_ = np.random.choice(range(len(next_states)), 1, weights)[0]
    # print(si_)
    # print(weights[si_])
    
    return next_states[si_]


def plotter(spot):
    #x = [1,2,3]
    fig = plt.figure(figsize = (L,W))
    ax = fig.add_subplot(1,1,1)
    spot_x_y = []
    for i in spot:
        spot_x_y += [(i[0]+0.5,i[1]+0.5)]
        ax.arrow(i[0]+0.5,i[1]+0.5, 0.5*np.sin(i[2]*np.pi/6),0.5*np.cos(i[2]*np.pi/6), head_width = 0.1, head_length = 0.2, fc = 'k', ec = 'k')
    
    spot_x_y = np.array(spot_x_y)
    
    #plot path
    plt.plot(*spot_x_y.T, color = 'b')
    
    #plot robot + orientation
    for i in spot:
        plt.plot(i[0]+0.5, i[1]+0.5, marker=(3, 0, 0), markersize=20,markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
        
    for i in range(0,8):
        for j in range(0,8):
            if (i == 0 or i == 7 or j == 0 or j == 7):
                ax.add_patch(plt.Rectangle((i,j), width = 1, height = 1, angle=0.0, fill = True, facecolor = 'r'))
            if (i == 3 and (j == 4 or j ==5 or j ==6)):
                ax.add_patch(plt.Rectangle((i,j), width = 1, height = 1, angle=0.0, fill = True, facecolor = 'y',  alpha = 0.8))
            if (i == 5 and j ==6):
                ax.add_patch(plt.Rectangle((i,j), width = 1, height = 1, angle=0.0, fill = True, facecolor = 'g', alpha = 0.7))
    plt.xlim([0,W])
    plt.ylim([0,L])
    plt.grid()
    plt.savefig("fig.jpg", dpi = 300)
    plt.show()


def inital_policy():
    pi_0 = {}
    tgt = (5, 6)

    for s in states:

        if s[0] == tgt[0] and s[1] == tgt[1]:
            a0 = 0
            a1 = 0

        else:
            tgt_direction = (tgt[0]-s[0], tgt[1]-s[1])
            h_angle = np.deg2rad(30*s[2])
            my_direction = (np.sin(h_angle), np.cos(h_angle))

            if my_direction[0]*tgt_direction[0] + my_direction[1]*tgt_direction[1] >= 0:
                a0 = +1
            else:
                a0 = -1

            hr_angle = np.deg2rad(30*(s[2]+1))
            hl_angle = np.deg2rad(30*(s[2]-1))

            l_direction = (np.sin(hl_angle), np.cos(hl_angle))
            r_direction = (np.sin(hr_angle), np.cos(hr_angle))

            l_dotprod = l_direction[0]*tgt_direction[0] + l_direction[1]*tgt_direction[1]
            r_dotprod = r_direction[0]*tgt_direction[0] + r_direction[1]*tgt_direction[1]

            if l_dotprod > r_dotprod:
                a1 = -1
            elif l_dotprod < r_dotprod:
                a1 = +1
            else:
                a1 = 0

        pi_0[s] = (a0, a1)

    return pi_0


# globals
gamma = 0.9
L = 8
W = 8
p_error = 0.25

if __name__ == "__main__":

    # 1a
    states = get_states()
    print("Number of states:", len(states))
    print()

    # 1b
    actions = get_actions()
    print("Number of actions:", len(actions))
    print()

    # 1c
    for i in range(5):
        s = choice(states)
        a = choice(actions)
        s_ = next_state_draw(s, a)

        print("P(",s,",",a,",",s_,"):", P_function(s, a, s_))
    
    print()

    # 1d
    for i in range(5):
        s = choice(states)
        a = choice(actions)

        print("s =",s,", a =",a,": s_ =",s_, next_state_draw(s, a))

    print()

    # 2a
    for i in range(5):
        s_ = choice(states)
        print("R(", s_, "):",R_function(None, None, s_))

    print()

    # mdp = MDP(states, actions, P_function, R_function, gamma)
    # pi_star = mdp.value_iteration()

    # spot = np.array([(1,1,1),(1,2,2),(1,3,3),(2,3,5,),(2,4,10),(3,4,12),(3,5,4)])
    # plotter(spot)
