import park
from rl.policy_network import ActorCritic
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 10000000000000
max_episodes = 10

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
				break

def a2c(env):
	num_inputs = env.observation_space.shape[0]
	num_outputs = env.action_space.n

	actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
	ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

	all_lengths = []
	average_lengths = []
	all_rewards = []
	entropy_term = 0

	for episode in range(max_episodes):
		log_probs = []
		values = []
		rewards = []

		state = env.reset()
		for steps in range(num_steps):
			value, policy_dist = actor_critic.forward(state)
			value = value.detach().numpy()[0,0]
			dist = policy_dist.detach().numpy() 

			action = np.random.choice(num_outputs, p=np.squeeze(dist))
			#print(action)
			log_prob = torch.log(policy_dist.squeeze(0)[action])
			entropy = -np.sum(np.mean(dist) * np.log(dist))
			new_state, reward, done, _ = env.step(action)

			rewards.append(reward)
			values.append(value)
			log_probs.append(log_prob)
			entropy_term += entropy
			state = new_state

			if done or steps == num_steps-1:
				Qval, _ = actor_critic.forward(new_state)
				Qval = Qval.detach().numpy()[0,0]
				all_rewards.append(np.sum(rewards))
				all_lengths.append(steps)
				average_lengths.append(np.mean(all_lengths[-10:]))
				if episode % 1 == 0:                    
					print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
				break

			 # compute Q values
		Qvals = np.zeros_like(values)
		for t in reversed(range(len(rewards))):
			Qval = rewards[t] + GAMMA * Qval
			Qvals[t] = Qval

		#update actor critic
		values = torch.FloatTensor(values)
		Qvals = torch.FloatTensor(Qvals)
		log_probs = torch.stack(log_probs)


		advantage = Qvals - values
		actor_loss = (-log_probs * advantage).mean()
		critic_loss = 0.5 * advantage.pow(2).mean()
		ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

		ac_optimizer.zero_grad()
		ac_loss.backward()
		ac_optimizer.step()

#env = park.make('cache')
#a2c(env)
env = park.make('cache')
admitAll(env)