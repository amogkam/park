import park
from rl.policy_network import PolicyNetwork, update_policy
import numpy as np

env = park.make('cache')
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

max_episode_num = 100
max_steps = 100000
numsteps = []
avg_numsteps = []
all_rewards = []

for episode in range(max_episode_num):
    state = env.reset()
    log_probs = []
    rewards = []

    for steps in range(max_steps):
        #env.render()
        action, log_prob = policy_net.get_action(state)
        new_state, reward, done, _ = env.step(action)
        #print(action)
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            update_policy(policy_net, rewards, log_probs)
            numsteps.append(steps)
            avg_numsteps.append(np.mean(numsteps[-10:]))
            all_rewards.append(np.sum(rewards))
            if episode % 1 == 0:
                print("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
            break
        
        state = new_state
