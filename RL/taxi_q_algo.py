import gym
import numpy as np
import random
from IPython.display import clear_output


#OpenAI Gyem Q-Learning module.
# Init Taxi-V2 Env

class QLearn():
    def __init__(self, aienv="Taxi-v2", a=0.1,g=0.6,e=0.1,learn_epochs=100001):
        # Initiliaze our environment to report back observations and rewards from the environment
        # This will be needed to know if we are making meaningful movements in the simulation.
        # This also allows us to see if the simulation is through or not.
        self.env = gym.make(aienv).env

        # Init arbitary values for Q table.  This is the matrix that will keep track of state to action mapping
        # based on points for each action at each state.
        print([self.env.observation_space.n, self.env.action_space.n])
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

        # Since the class gives the option to run epochs individually, the state variable will be accessed throughout
        # entire class in both the run_epoch method and the run_to_completion method
        self.state = self.env.reset()

        # Epoch will count the number of learning iterations the object has undergone.  Penalties will count
        # the number of times that significantly bad moves were taken
        self.epochs, self.penalties, self.reward = 0, 0, 0
        # This will tell the learner if the agent has completed the task at any given step
        self.done = False

        # number of steps to complete a batch learn session
        self.l_ep = learn_epochs



    def take_step(self, render=False):
        # beware of render method as it causes the program to learn much much slower
        if(render):
            self.env.render()

        # Multiarm bandit
        # exploration 10 percent of the time, exploitation 90 percent of the time
        if random.uniform(0, 1) < self.epsilon:
            # Check the action space
            self.action = self.env.action_space.sample()
        else:
            # Check the learned values
            self.action = np.argmax(self.q_table[self.state])
        
        # Take the action chosen and observe the reward.  Also report if the simulation completed on theis step
        # And indicate the next state that we stepped to.
        self.next_state, self.reward, self.done, self.info = self.env.step(self.action)
        
        # Perform Q-Learning Calculation.  Currently the Equation is based off of a finite horizon scheme
        # This will be tweaked later to allow for finite or infinite horizon
        self.old_value = self.q_table[self.state, self.action]
        self.next_max = np.max(self.q_table[self.next_state])

        # Update the new value
        self.new_value = (1 - self.alpha) * self.old_value + self.alpha * \
            (self.reward + self.gamma * self.next_max)
        self.q_table[self.state, self.action] = self.new_value

        # Add up penalties to report as final score
        if self.reward == -10:
            self.penalties += 1

        self.state = self.next_state
        self.epochs += 1

        

    def run_epoch(self):
        self.state = self.env.reset()

        # Init Vars
        self.epochs, self.penalties, self.reward = 0, 0, 0
        self.done = False

        while not self.done:
            self.take_step()

    def learn(self):
        # The program will execute a number of epochs to solve the problem.  Each iteration, the algorithm will solve the 
        # problem incrementally better as the Q Table becomes more optimized for the task at hand
        for i in range(1, self.l_ep):
            self.take_step()
            if i % 1000 == 0:
                clear_output(wait=True)
                print("Episode: ", i)
                print("penalties: ", self.penalties)
                print("epochs: ", self.epochs)
            




def main():
    taxi_agent = QLearn("Taxi-v2")
    taxi_agent.learn()
    
        




# print("Training finished.")


if __name__ == "__main__":
    main()