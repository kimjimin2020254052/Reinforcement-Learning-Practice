import rclpy
from rclpy.node import Node
import gymnasium as gym
import numpy as np
import time

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_actions))
        # parameter setup
        self.alpha = alpha            # learning rate 
        self.gamma = gamma            # discounting rate
        self.epsilon = epsilon        # ratio of eploration and exploitation
        self.epsilon_decay = 0.00033   # epsilon decrease

    def get_action(self, state, epsilon):
        random = np.random.random()

        if random <= float(1-epsilon):
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, 6)
        return action

    def update(self, s, a, r, s_next):
        self.q_table[s][a] += self.alpha*(r + self.gamma*np.max(self.q_table[s_next])-self.q_table[s][a])

class TaxiNode(Node):
    def __init__(self):
        super().__init__('taxi_node')
        self.agent = QLearningAgent(n_states=500, n_actions=6)
        self.env = gym.make('Taxi-v3', render_mode="ansi")
        
    
    def run_training(self, max_episodes):

        for episode in range(max_episodes):
            # env.reset() provides two values. One is current state and info(we don't need now)
            state, _ = self.env.reset()
            done = False

            while not done and rclpy.ok(): 
                action = self.agent.get_action(state, self.agent.epsilon)
                
                # env.step(action) provides five values. 
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.agent.update(state, action, reward, next_state)
                state = next_state
                
            # initialization
            if episode % 500 == 0:
                print(self.agent.q_table)
            self.agent.epsilon -= self.agent.epsilon_decay
        
        
        self.env.close()
        self.env = gym.make('Taxi-v3', render_mode="human")
        done = False
        state, _ = self.env.reset()
        while not done:
            action = self.agent.get_action(state, 0)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            state = next_state
            time.sleep(0.5)
        self.env.close()
           
                    


def main(args=None):
    rclpy.init(args=args)
    node = TaxiNode()    

    node.run_training(max_episodes=3000)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()