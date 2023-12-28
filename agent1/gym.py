import gym
from gym import spaces
import numpy as np
from sklearn.linear_model import SGDClassifier
from joblib import dump
import os

# Custom Gym Environment
class ClassificationEnv(gym.Env):
    def __init__(self, data, labels):
        super(ClassificationEnv, self).__init__()
        self.data = data
        self.labels = labels
        self.current_index = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Two actions: 0 for 'not annoying', 1 for 'annoying'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def step(self, action):
        # Check if the action is correct
        reward = 1 if action == self.labels[self.current_index] else -1

        self.current_index += 1
        done = self.current_index == len(self.labels)
        next_state = self.data[self.current_index % len(self.data)]  # Loop over data

        return next_state, reward, done, {}

    def reset(self):
        self.current_index = 0
        return self.data[0]

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 5)  # 100 samples, 5 features each
y = np.random.randint(0, 2, 100)  # Random binary labels

# Initialize environment and classifier
env = ClassificationEnv(X, y)
classifier = SGDClassifier(loss='log')  # Using logistic regression for binary classification

# Variables for controlling learning and inference
iteration = 0
inference_start_iteration = 100

# Directory for saving models
model_directory = "models"
os.makedirs(model_directory, exist_ok=True)

# Continuous learning and inference loop
while True:
    # Get current state
    state = env.reset()
    done = False

    while not done:
        if iteration >= inference_start_iteration:
            # Perform inference after 100 iterations
            action = classifier.predict([state])[0]
        else:
            # Random action or some default action before 100 iterations
            action = env.action_space.sample()
        
        # Step through the environment
        next_state, reward, done, info = env.step(action)

        # Continuous learning: update the model with new data (state, action)
        classifier.partial_fit([state], [action], classes=np.unique(y))
        
        # Update the state
        state = next_state

    iteration += 1

    # Save the model every 100 iterations
    if iteration % 100 == 0:
        model_path = os.path.join(model_directory, f'classifier_model_iteration_{iteration}.joblib')
        dump(classifier, model_path)

    # Optional: Break after a certain number of iterations
    # if iteration == some_upper_limit:
    #     break
