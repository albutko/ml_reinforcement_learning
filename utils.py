import numpy as np
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def policy_eval(policy, env, discount_factor=0.5):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    E, T, R = _encode_policy_matrices(policy, env)
    V = np.linalg.solve(np.eye(env.nS) - discount_factor*np.matmul(E, T), np.matmul(E, R))

    return V

def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 4 arguments:
            policy, env, reward, discount_factor.
        reward: Reward for being in each state
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    deltas=list()

    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    last_pol_values = np.zeros(env.env.nS)
    i = 0
    while True:
        i +=1
        print(f'Epsiode: {i}')
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  #eval current policy
        policy_stable = True  #Check if policy did improve (Set it as True first)
        for state in range(env.env.nS):  #for each states
            chosen_act = np.argmax(policy[state])  #best action (Highest prob) under current policy
            act_values = one_step_lookahead(state, curr_pol_val)  #use one step lookahead to find action values
            best_act = np.argmax(act_values) #find best action
            if chosen_act != best_act:
                # print('State: %d; Chosen action: %d; best_act: %d, '%(state, chosen_act, best_act))
                policy_stable = False  #Greedily find best action
            policy[state] = np.eye(env.env.nA)[best_act]  #update

        deltas.append(np.sqrt(np.sum(np.square(curr_pol_val-last_pol_values))))
        last_pol_values = np.copy(curr_pol_val)
        if policy_stable:
            return policy, curr_pol_val, deltas



    return policy, np.zeros(env.env.nS)

def value_iteration(env, theta=0.1, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for act in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][act]:
                A[act] += prob * (reward + discount_factor*V[next_state])
        return A

    V = np.zeros(env.env.nS)
    last_V = np.zeros(env.env.nS)
    i = 0
    deltas = list()
    while True:
        i +=1
        print(f'Epsiode: {i}')
        delta = 0  #checker for improvements across states
        for state in range(env.env.nS):
            act_values = one_step_lookahead(state,V)  #lookahead one step
            best_act_value = np.max(act_values) #get best action value
            delta = max(delta,np.abs(best_act_value - V[state]))  #find max delta across all states
            V[state] = best_act_value  #update value to best action value
        deltas.append(np.sqrt(np.sum(np.square(V-last_V))))
        last_V = np.copy(V)
        if delta < theta:  #if max improvement less than threshold
            break
    policy = np.zeros([env.env.nS, env.env.nA])
    for state in range(env.env.nS):  #for all states, create deterministic policy
        act_val = one_step_lookahead(state,V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1

    return policy, V, deltas


def q_learn(env, alpha=0.1, gamma=0.6, epsilon=0.4, episodes=1000, optimistic_val=1.0, exploration='optimistic'):
    # Optimistic randomization
    if exploration == 'optimistic':
        q_table = np.full((env.env.nS,env.env.nA), optimistic_val)
    else:
        q_table = np.zeros([env.env.nS, env.env.nA])

    visit_count = np.zeros(env.env.nS)

    A = env.env.nA
    steps = 0
    last_Q = q_table

    epochs = 0
    steps_per_episode = list()
    avg_reward_per_epi = list()
    time_by_step = list()
    max_epsilon = epsilon
    decay_rate = np.log(0.01/epsilon)/-episodes
    deltas = list()

    for i in range(1, episodes):
        steps += epochs
        state = env.reset()
        epochs, reward, = 0, 0
        done = False
        rewards = list()
        epsilon = max_epsilon*np.exp(-1*decay_rate*i)
        start = timer()
        while not done:

            if exploration == 'random':
                action = np.random.randint(A)

            elif exploration == 'optimistic':
                action = np.argmax(q_table[state])

            elif exploration == 'epsilon':
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action)
            # print('next state: %d, reward: %d, done: %s' % (next_state, reward, done))
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            q_table[state, action] = new_value
            state = next_state


            if done:
                action = np.argmax(q_table[state]) # Exploit learned values
                next_state, after_reward, _, info = env.step(action)
                # print('next state: %d, reward: %d, done: %s' % (next_state, reward, done))
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (after_reward + gamma*next_max)

                q_table[state, action] = new_value

            epochs += 1
            rewards.append(reward)
        time_by_step.append(timer()-start)
        if i % 10000 == 0:
            print(f"Episode: {i}")
        delt = np.sqrt(np.sum((last_Q-q_table)**2))
        deltas.append(delt)
        last_Q = q_table.copy()

        avg_reward_per_epi.append(np.mean(rewards))
        steps_per_episode.append(epochs)


    return q_table, deltas, steps_per_episode, avg_reward_per_epi, time_by_step


def _encode_policy_matrices(policy, env):
    """ Encode policy into 3 matrices for matrix solution of value
    Inputs:
        policy
        env
    outputs:
        E (AxS*A)
        T (S*AxA)
        R (S*A,)
    """
    A = env.env.nA
    S = env.env.nS
    E = np.zeros((S, S*A))
    T = np.zeros((S*A, S))
    R = np.zeros(S*A)
    for i, state in enumerate(policy):
        E[i,i*A+np.argmax(state)] = 1
        for j in range(A):

            for pos in env.P[i][j]:
                prob, next_state, reward, done = pos
                T[i*A+j, next_state] = prob
                R[i*A+j] += prob*reward

    return (E, T, R)

def protocol_from_q(Q):
    A = np.eye(Q.shape[1])
    optimal_actions = np.argmax(Q, axis=1)
    return A[optimal_actions,:]

def grid_from_policy(policy, env):
    map = list()
    print('|',end=" ")
    for i, s in enumerate(policy):
        action = np.argmax(s)
        if env.desc[i//8,i%8] == b'H':
            map.append('O')
        elif env.desc[i//8,i%8] == b'G':
            map.append('G')
        else:
            if action == 0:
                map.append('<')
            elif action == 1:
                map.append('v')
            elif action == 2:
                map.append('>')
            elif action == 3:
                map.append('^')
            else:
                map.append('-')
    for i, state in enumerate(map):
        if i%8 == 0 and i != 0:
            print('|\n|', end=" ")
        print(state, end=" ")
    print('|')


def policy_map(policy, values, env, title=''):
    map = list()
    for i, s in enumerate(policy):
        action = np.argmax(s)
        if env.desc[i//8,i%8] == b'H':
            map.append('O')
        elif env.desc[i//8,i%8] == b'G':
            map.append('G')
        else:
            if action == 0:
                map.append('<')
            elif action == 1:
                map.append('v')
            elif action == 2:
                map.append('>')
            elif action == 3:
                map.append('^')
            else:
                map.append('-')

    reshaped_values = np.reshape(values, (8,8))
    fig, ax = plt.subplots()
    im = ax.imshow(reshaped_values)

    for i, state in enumerate(map):
        ax.text(i%8, i//8, map[i], ha="center", color="w")
    fig.colorbar(im)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)

def policy_map_q(policy, q_values, env, title='',file=''):
    values = list()
    map = list()
    for i, s in enumerate(policy):
        action = np.argmax(s)
        values.append(q_values[i][action])
        if env.desc[i//8,i%8] == b'H':
            map.append('O')
        elif env.desc[i//8,i%8] == b'G':
            map.append('G')
        else:
            if action == 0:
                map.append('<')
            elif action == 1:
                map.append('v')
            elif action == 2:
                map.append('>')
            elif action == 3:
                map.append('^')
            else:
                map.append('-')
    values = np.array(values)
    reshaped_values = np.reshape(values, (8,8))
    fig, ax = plt.subplots()
    im = ax.imshow(reshaped_values)

    for i, state in enumerate(map):
        ax.text(i%8, i//8, map[i], ha="center", color="w")
    fig.colorbar(im)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if file != '':
        plt.savefig(file, bbox_inches='tight')
    

def run_policy(policy, env, iters=100):
    rewards = list()
    steps_per_episode = list()
    for _ in range(iters):
        reward_sum = 0
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.argmax(policy[state])
            next_state, reward, done, info = env.step(action)
            steps +=1
            reward_sum += reward
            state = next_state

        rewards.append(reward_sum)
        steps_per_episode.append(steps)
    return rewards, steps_per_episode


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
