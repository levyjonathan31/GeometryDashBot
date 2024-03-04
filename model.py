import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keyboard
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from geometry_dash_env import env
from noisy_layer import NoisyDense
import time
# from sliding_objects import FrameStacker

from config import screen
train = True
training_episodes = 10000
seed = 42
gamma = 0.99  # Discount factor for past rewards
omega = 0.5  # How much prioritization is used
n_steps = 10  # Number of steps to look ahead
epsilon = 0.0  # Epsilon greedy parameter
epsilon_min = 0.0  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken

batch_size = 64  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_actions = 2

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(108, 192, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = NoisyDense(512, activation="relu")(layer4)
    action = NoisyDense(num_actions, activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)
model = None
model_target = None
try:
    model = load_model('model.keras', custom_objects={'NoisyDense': NoisyDense})
    model_target = load_model('model_target.keras', custom_objects={'NoisyDense': NoisyDense})
    print("Using pre-trained models")
except:
    model = create_q_model()
    model_target = create_q_model()
    print("Creating new models")


optimizer = keras.optimizers.Adam(learning_rate=0.0000625, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_reward = 0
episode_count = 0
frame_count = 0

# Prioritized replay probabilities
prioritized_replay = []

# Number of frames to take random action and observe output
epsilon_random_frames = 7500.0
# Number of frames for exploration
epsilon_greedy_frames = 150000.0
# Maximum replay length
max_memory_length = 100000
update_after_actions = 10
# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

pressed_q = False
if train:   
    env = env()
    while not pressed_q and episode_count < training_episodes:
        # if press q, then save the model
        if keyboard.is_pressed('q'):      
            pressed_q = True
        state = env.reset()
        if env.done: 
            continue
        episode_reward = 0
        for timestep in range(1, max_steps_per_episode):
            start_time = time.time()
            if keyboard.is_pressed('q'):
                pressed_q = True
                break
            frame_count += 1
            if frame_count % 1000 == 0:
                print("Frame count: ", frame_count)
            # Use epsilon-greedy for exploration
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=train)
            action = 0
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                # Take best action
                action = tf.argmax(action_probs[0]).numpy() 

            
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment           
            state_next, reward, done, _ = env.step(action, episode_reward)
            state_next = np.array(state)
            if not done:
                episode_reward = reward
            # q_value = action_probs[0][action].numpy()
            
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)

            state = state_next
            # for n-step return
            multi_step_return = 0
            if len(rewards_history) >= n_steps and timestep >= n_steps:
                discount_factors = np.array([gamma**i for i in range(n_steps)])
                # Get the n-step return
                multi_step_return = np.sum(discount_factors * rewards_history[-n_steps:])
                #expand dims for prediction of target model, not convert to tensor
                state_tensor_n = tf.expand_dims(state_history[-n_steps], 0)
                multi_step_return += gamma**n_steps * model_target.predict(state_tensor_n).max(1).item() * (1 - done_history[-n_steps])
                # Calculate TD error for multi-step return
                q_value_n_steps_ago = model.predict(tf.expand_dims(state_history[-n_steps], 0))[0][action_history[-n_steps]]
                # q_value = action_probs[0][action].numpy()
                temporal_difference_error = abs(multi_step_return - q_value_n_steps_ago)

                # Update priority
                prioritized_replay.append(temporal_difference_error**omega)
                if len(prioritized_replay) > max_memory_length:
                    del prioritized_replay[:1]
            # Update every 10th frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size and len(rewards_history) > n_steps:

                # Implement Prioritized Experience Replay
                sample_probabilities = prioritized_replay / (np.sum(prioritized_replay) + 1e-10)
                indices = np.random.choice(range(len(prioritized_replay)), size=batch_size, p = sample_probabilities)
                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Use multi-step return as the updated Q-values
                updated_q_values = multi_step_return
                # updated_q_values = rewards_sample + gamma * tf.reduce_max(
                #     future_rewards, axis=1
                # )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)
    
                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                print("Time taken: ", time_taken)
            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break
            time_taken = time.time() - start_time


        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 2000:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        print("Episode {} reward: {}".format(episode_count, episode_reward))
        print("End timestep: ", timestep)
        episode_count += 1  

    # save the models
    model.save('model.keras')
    model_target.save('model_target.keras')
    # plot the running rewards
    plt.plot(episode_reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('running_reward.png')
    plt.show()

        
        # if running_reward > 40:  # Condition to consider the task solved
        #     print("Solved at episode {}!".format(episode_count))
        #     break
if not train:
    env = env()
    timestep_total = 0 
    i = 0
    score = 0
    while i < 20:
        state = env.reset()
        if env.done: 
            continue
        else:
            print("Attempt 1: ", i+1)
            for timestep in range(1, max_steps_per_episode):
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model_target(state_tensor, training=train)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                state, reward, done, _ = env.step(action, score)
                score += reward
                if done:
                    print("Ended at timestep: ", timestep)
                    timestep_total += timestep
                    i += 1
                    break
    print("Average timestep: ", timestep_total/20)