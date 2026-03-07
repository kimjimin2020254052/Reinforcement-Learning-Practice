#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self): 
        # Get some parents data
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, s):
        q = self.fc1(s) 
        q = torch.relu(q)
        Q = self.fc2(q)
        return Q


class DQNAgent:
    def __init__(self):
        self.model = DQN().to('cuda')
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)  # learning late lr.
        self.epsilon = 1.0
        self.epsilon_decay = 0.001
        self.gamma = 0.99

    def get_action(self, state, epsilon):
        random = np.random.random()

        if random >= float(epsilon):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to('cuda')
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = q_values.argmax().item()
            
        else:
            action = np.random.randint(0, 2)
        return action

    def memory_store(self, s, a, r, s_next, done):
        self.memory.append((s,a,r,s_next,done))

    def train_model(self):
        if len(self.memory) < 32 : return
        mini_batch = random.sample(self.memory, 32)

        # reput in the new [ ]. We have to make new list into torch. And put all parameters seperately
        states      = torch.FloatTensor(np.array([x[0] for x in mini_batch])).to('cuda')
        actions     = torch.LongTensor([x[1] for x in mini_batch]).view(-1, 1).to('cuda')
        rewards     = torch.FloatTensor([x[2] for x in mini_batch]).view(-1, 1).to('cuda')
        next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch])).to('cuda')
        dones       = torch.FloatTensor([x[4] for x in mini_batch]).view(-1, 1).to('cuda')

        # predicted q
        Qvalues = self.model(states)
        predicted_q = Qvalues.gather(1, actions)
        # traget q
        Qvalues_next = self.model(next_states)
        target_max_q = Qvalues_next.max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma*target_max_q*(1-dones)


        # Loss calculator
        loss = F.mse_loss(predicted_q, target_q)
        
        # Adam 3 combo
        self.optimizer.zero_grad()   # clean the garbage
        loss.backward()              # Backpropagation to evaluate gradients of each W, b
        self.optimizer.step()        # It can change W, b values considering Momentum and RMSprop(use learning rate)


class CartPoleNode(Node):
    def __init__(self):
        super().__init__('cart_pole_node')
        # model.parameter() => it can send all parameter data to Adam. 
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.agent = DQNAgent()
    
    def run_training(self, max_episodes):

        for episode in range(max_episodes):
            # env.reset() provides two values. One is current state and info(we don't need now)
            state, _ = self.env.reset()
            done = False
            score = 0
            while not done and rclpy.ok(): 
                action = self.agent.get_action(state, self.agent.epsilon)
                
                # env.step(action) provides five values. 
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.memory_store(state, action, reward, observation, done)
                self.agent.train_model()
                score += reward
                state = observation

            self.agent.epsilon = max(0.01, self.agent.epsilon-self.agent.epsilon_decay)
            print(f"episode : {episode}, score : {score}")
        # finish
        self.env.close()
           
                    


def main(args=None):
    rclpy.init(args=args)
    node = CartPoleNode()    

    node.run_training(max_episodes=3000)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()