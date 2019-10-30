import park
import numpy as np

num_steps = 10000000000000
max_episodes = 1

class AdmitAllAgent(object):
    def predict(self, obs):
        return 1, None

def admitAll(env):
	all_lengths = []
	average_lengths = []
	all_rewards = []

	agent = AdmitAllAgent()

	for episode in range(max_episodes):
		rewards = []
		state = env.reset()
		for steps in range(num_steps):
			action, _ = agent.predict(state)
			new_state, reward, done, _ = env.step(action)

			rewards.append(reward)
			state = new_state

			if done or steps == num_steps-1:
				all_rewards.append(np.sum(rewards))
				all_lengths.append(steps)
				average_lengths.append(np.mean(all_lengths[-10:]))

				if episode % 1 == 0:                    
					print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
					print("count_bhr: {}, count_ohr: {}, total_size: {}, bhr_ratio: {} \n".format(env.sim.count_bhr, env.sim.count_ohr, env.sim.size_all, float(float(env.sim.count_bhr) / float(env.sim.size_all))))
				break

env = park.make('cache')
admitAll(env)


