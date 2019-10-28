import park

env = park.make('cache')

obs = env.reset()
print(obs)
done = False

while not done:
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
    #print(reward)
