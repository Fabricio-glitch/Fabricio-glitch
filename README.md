import numpy as np
import random

# Definindo o ambiente (uma grade 5x5)
class Environment:
    def __init__(self, grid_size=5, goal_position=(4, 4)):
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.reset()

    def reset(self):
        # Posição inicial do agente
        self.agent_position = [0, 0]
        return self.agent_position

    def step(self, action):
        # Define as ações: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:
            self.agent_position[1] += 1

        # Recompensa e fim do episódio
        if self.agent_position == list(self.goal_position):
            reward = 1  # Recompensa por alcançar o objetivo
            done = True
        else:
            reward = -0.1  # Penalidade por cada movimento
            done = False

        return self.agent_position, reward, done

# Algoritmo Q-Learning
class QLearningAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.environment = environment
        self.q_table = np.zeros((environment.grid_size, environment.grid_size, 4))  # Tabela Q
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            # Escolhe ação aleatória (exploração)
            return random.randint(0, 3)
        else:
            # Escolhe ação baseada na tabela Q (exploração)
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.q_table[state[0], state[1], action]
        max_future_q_value = np.max(self.q_table[next_state[0], next_state[1]])

        # Fórmula de atualização do Q-Learning
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                      self.learning_rate * (reward + self.discount_factor * max_future_q_value)

        self.q_table[state[0], state[1], action] = new_q_value

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.environment.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.environment.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            # Decai a taxa de exploração
            self.exploration_rate *= self.exploration_decay

# Criando o ambiente e o agente
env = Environment()
agent = QLearningAgent(env)

# Treinando o agente
agent.train(episodes=1000)

# Testando o agente após o treinamento
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, _, done = env.step(action)
    print(f"Estado atual: {state}")

print("Objetivo alcançado!")
