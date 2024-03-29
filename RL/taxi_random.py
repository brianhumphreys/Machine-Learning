import gym
from time import sleep
from IPython import display

#Create the environment
env = gym.make("Taxi-v2").env

env.s = 328

# Setting the number of iterations, penalties and reward to zero,
epochs = 0
penalties, reward = 0, 0

frames = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into the dictionary for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    })

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

# Printing all the possible actions, states, rewards.
def printFrames(frames):
    for i, frame in enumerate(frames):
        
        # print(type(frame['frame']))
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        display.clear_output(wait=True)
        sleep(.1)
        
printFrames(frames)