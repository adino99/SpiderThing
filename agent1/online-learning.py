import zmq
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import json 
import redis
import time
from keras.layers import GRU

model_save_path = 'model_normal.h5'  # Specify your path here

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_data():

    # Get all keys
    all_keys = redis_client.keys('*')

    # Sort keys
    sorted_keys = sorted(all_keys, key=lambda x: x.decode())

    if len(sorted_keys) == 0:
        return []
    # Iterate and print or process sorted keys
    ret = []
    keys = []
    for key in sorted_keys:
        val = redis_client.get(key.decode())
        # print(val)
        try:
            arr_json = json.loads(val)
            # ret.append(float(arr_json['home']))
            ret.append([float(arr_json['decibel']), float(arr_json['ambient']), float(arr_json['home'])])
            keys.append(key.decode())
        except:
            pass
    return ret, keys

# Hyperparameters
learning_rate = 0.0001
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 2        # Number of data buffer before learning new model
sequence_size = 100     # Number of fixed-sized features as input trainig
feature_size = 3        # Number of feature readings in sequence
action_size = 2         # Two possible actions (Annoying, Not-Annoying)


# DQN Agent
class DQNAgent:
    def __init__(self, sequence_size, feature_size, action_size, memory_size=30):
        self.sequence_size = sequence_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Adding a GRU layer - you may need to adjust the number of units
        model.add(GRU(sequence_size, input_shape=(sequence_size, feature_size), activation='relu'))
        model.add(Dense(64, activation='relu'))  # You can still use Dense layers after GRU
        model.add(Dense(32, activation='relu'))  # You can still use Dense layers after GRU
        model.add(Dense(8, activation='relu'))  # You can still use Dense layers after GRU
        # model.add(Dense(self.action_size, activation='softmax'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def act(self, sequence):
        print("Epsilon:", epsilon)
        if np.random.rand() <= epsilon:
            return (random.randrange(self.action_size), None, False)
        act_values = self.model.predict(sequence)
        return (np.argmax(act_values[0]), act_values, True)

    def remember(self, sequence, action, reward, next_sequence, done):
        self.memory.append((sequence, action, reward, next_sequence, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for sequence, action, reward, next_sequence, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(self.model.predict(next_sequence)[0])
            target_f = self.model.predict(sequence)
            target_f[0][action] = target
            self.model.fit(sequence, target_f, epochs=1, verbose=0)
        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    def save_model(self, filepath):
        self.model.save(filepath)

# ZMQ Communication
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

# Initialize the DQN Agent
agent = DQNAgent(sequence_size, feature_size, action_size, batch_size)

# State Initialization
state = deque(maxlen=sequence_size)

# Infinite Online Learning Loop
while True:
    # Simulate receiving a new decibel reading
    # new_decibel_level = np.random.normal(60, 10)
    state, keys = get_data()
    # state.append(new_decibel_level)
    # print(state)
    # time.sleep(5)
    if len(state) < sequence_size * 2:
        print("State not enough:", state)
    elif len(state) >= sequence_size * 2:
        data_current_state = state[-sequence_size*2:-sequence_size]
        data_current_keys = keys[-sequence_size*2:-sequence_size]
        data_next_state = state[-sequence_size:]
        
        # print("Keys:", data_current_keys)
        # print("Cur:", data_current_state)
        # print("Next:", data_next_state)

        current_state = np.array([list(data_current_state)])
        action, act_values, is_result_from_inference = agent.act(current_state)

        # Package action and state into JSON
        data_to_send = json.dumps({
            "action": int(action),
            "data_sequence": list(data_current_state)
        })

        # print("Data to Send:", data_to_send)
        # Send action to Agent 2 (Evaluator)
        socket.send_string(data_to_send)
        reward = socket.recv_pyobj()

        # # Simulate next state
        # next_state = deque(state, maxlen=sequence_size)
        # next_state.append(np.random.normal(60, 10))
        next_state = np.array([list(data_next_state)])
        done = False  # Assuming continuous task

        # Remember and replay
        agent.remember(current_state, action, reward, next_state, done)
        if is_result_from_inference:
            print("Agent Memory:", len(agent.memory), "Act Values:", act_values)

        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
            agent.memory = deque(maxlen=batch_size)
            # TODO: Write as new saved model
            agent.save_model(model_save_path)
