import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keyboard
from tensorflow import keras
from tensorflow.keras import layers

from geometry_dash_env import env

# from sliding_objects import FrameStacker

from config import screen
training = False 
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.02  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken

batch_size = 32  # Size of batch taken from replay buffer
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

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)

model = create_q_model()

model_target = create_q_model()

try:
    model = keras.models.load_model('model.keras')
    model_target = keras.models.load_model('model_target.keras')
except:
    pass

optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Number of frames to take random action and observe output
epsilon_random_frames = 500.0
# Number of frames for exploration
epsilon_greedy_frames = 10000.0
# Maximum replay length
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

pressed_q = False
if training:
    env = env()
    while not pressed_q:
        # if press q, then save the model
        if keyboard.is_pressed('q'):      
            pressed_q = True
        state = env.reset()
        if env.done: 
            continue
        episode_reward = 0
        for timestep in range(1, max_steps_per_episode):
            if keyboard.is_pressed('q'):
                pressed_q = True
                break
            frame_count += 1
            if frame_count % 1000 == 0:
                print("Frame count: ", frame_count)
            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=True)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy() 

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment           
            state_next, reward, done, _ = env.step(action)

            state_next = np.array(state)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            rewards_history[-15:] = [x * 1.3 for x in rewards_history[-10:]]
            if (done):
                rewards_history[-15:] = [x - 2 for x in rewards_history[-10:]]
            state = state_next

            # Update every 5th frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)
                # print([arr.shape for arr in state_next_history])
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
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

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

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
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
if not training:
    env = env()
    timestep_total = 0 
    i = 0
    while i < 20:
        state = env.reset()
        if env.done: 
            continue
        else:
            print("Attempt 1: ", i+1)
            for timestep in range(1, max_steps_per_episode):
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                state, _, done, _ = env.step(action)
                if done:
                    print("Ended at timestep: ", timestep)
                    timestep_total += timestep
                    i += 1
                    break
    print("Average timestep: ", timestep_total/20)