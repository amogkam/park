import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import matplotlib.pyplot as plt

# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-1):
		super(PolicyNetwork, self).__init__()

		self.num_actions = num_actions
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, num_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		#print(x)
		x = F.softmax(self.linear2(x), dim=1)
		return x 
	
	def get_action(self, state):
		#print(np.ndarray(state))
		#print(torch.from_numpy(np.ndarray(state)))
		state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
		#print(state)
		probs = self.forward(Variable(state))
		#print(probs)
		highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
		log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
		return highest_prob_action, log_prob

class ActorCritic(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
		super(ActorCritic, self).__init__()

		self.num_actions = num_actions
		self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
		self.critic_linear2 = nn.Linear(hidden_size, 1)

		self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
		self.actor_linear2 = nn.Linear(hidden_size, num_actions)
	
	def forward(self, state):
		state = Variable(torch.from_numpy(np.asarray(state)).float().unsqueeze(0))
		value = F.relu(self.critic_linear1(state))
		value = self.critic_linear2(value)
		
		policy_dist = F.relu(self.actor_linear1(state))
		policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

		return value, policy_dist


def update_policy(policy_network, rewards, log_probs):
	discounted_rewards = []

	for t in range(len(rewards)):
		Gt = 0 
		pw = 0
		for r in rewards[t:]:
			Gt = Gt + GAMMA**pw * r
			pw = pw + 1
		discounted_rewards.append(Gt)
		
	discounted_rewards = torch.tensor(discounted_rewards)
	discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

	policy_gradient = []
	for log_prob, Gt in zip(log_probs, discounted_rewards):
		policy_gradient.append(-log_prob * Gt)
	
	policy_network.optimizer.zero_grad()
	policy_gradient = torch.stack(policy_gradient).sum()
	policy_gradient.backward()
	policy_network.optimizer.step()