import gym
import tensorflow as tf
import pdb

def define_q_network(STATE_DIM, NUM_ACTIONS):
	"""
		Defines a Q-network class for continuous state, discrete action
		RL environments, with state dimension and action count hardwired
		into the class.

		This wrapper is required in order to include input signatures for 
		methods of the Q-network class which are decorated with tf.function
	"""

	class QNetwork(tf.keras.Model):
		"""
			A fully connected neural network for modeling a Q-function
			in a RL environment with continuous state and discrete 
			actions.

			The network takes a state as input and outputs a vector of
			Q-value estimates for the state and each action.

			Supports batch computation.
		"""


		def __init__(self, hidden_dims, reg_coeff=0.01, lr=1e-4):
			"""
				Initializes a Q-network.

				Args:
					hidden_dims: a list of hidden layer dimensions (ints) which
						define the network structure.
					reg_coeff: regularization coefficient for network weights
					lr: learning rate for optimizer
			"""
			super().__init__()

			# define hidden layers
			self.network_layers = [
				tf.keras.layers.Dense(dim, activation='relu', 
					kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coeff, reg_coeff))
				for
				dim
				in
				hidden_dims
			]

			# define output layer
			self.network_layers.append(
				tf.keras.layers.Dense(NUM_ACTIONS, activation=None, 
					kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coeff, reg_coeff))
			)

			self.optimizer = tf.keras.optimizers.Adam(lr=lr)
		

		@tf.function(input_signature=[
			tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32)])
		def call(self, s):
			"""
				Computes a vector of Q-value estimates for each action
				for each state in a batch.

				Args:
					s: batch of states for which to estimate Q-values 
			"""
			intermediate = s
			for layer in self.network_layers:
				intermediate = layer(intermediate)

			return tf.reshape(intermediate, [tf.shape(s)[0], NUM_ACTIONS])	


		@tf.function(input_signature=[
			tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
			tf.TensorSpec(shape=[None], dtype=tf.int32), 
			tf.TensorSpec(shape=[None], dtype=tf.float32)])
		def train_step(self, s, a, target):
			"""
				Performs one optimization step for the network weights using
				MSE loss between Q-value estimates and provided targets.

				Args:
					s: a batch of states experienced by the agent
					a: the actions (ints) taken at each state in s
					target: a target value for the Q-value estimates
			"""
			with tf.GradientTape() as tape:
				loss = self.loss(s, a, target)
				gradients = tape.gradient(loss, self.trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
			return loss


		@tf.function(input_signature=[
			tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32),
			tf.TensorSpec(shape=[None], dtype=tf.int32), 
			tf.TensorSpec(shape=[None], dtype=tf.float32)])
		def loss(self, s, a, target):
			"""
				Computes the network loss for a batch as the mean-squared-error 
				between Q-value estimates and provided targets.

				Args:
					s: a batch of states experienced by the agent
					a: the actions (ints) taken at each state in s
					target: a target value for the Q-value estimates
			"""

			ind = tf.transpose(tf.stack([tf.range(tf.shape(s)[0]), a]))
			sample_losses = (tf.reduce_sum(self.call(s)*tf.one_hot(a, NUM_ACTIONS), axis=1)-target)**2

			return tf.reduce_mean(sample_losses)


		@tf.function(input_signature=[
		 	tf.TensorSpec(shape=[STATE_DIM], dtype=tf.float32)])
		def get_action(self, s):
			"""
				Computes the current best estimate of the optimal action for
				a batch of states by maximizing the Q-value for each over 
				all actions.

				Args:
					s: a batch of states for which to choose actions
			"""
			return tf.argmax(tf.reshape(self.call(tf.expand_dims(s, 0)), [-1]))


		@tf.function(input_signature=[
			tf.TensorSpec(shape=[None, STATE_DIM], dtype=tf.float32)])
		def value(self, states):
			"""
				Computes estimates of the values of a batch of states by
				maximizing the Q-value for each over all actions.

				Args:
					s: a batch of states for which to compute values
			"""
			return tf.math.reduce_max(self.call(states), axis=1)

	return QNetwork

if __name__ == '__main__':
	from collections import deque
	import numpy as np
	import random
	import matplotlib.pyplot as plt

	plt.ion()

	env = gym.make('CartPole-v1')
	CartPoleQNetwork = define_q_network(STATE_DIM=4, NUM_ACTIONS=2)

	### Network/training hyperparameters
	HIDDEN_DIMS = [20, 32, 30]

	DISCOUNT_RATE = 0.98
	UPDATE_TARGET_NET_PERIOD = 100 
	INIT_REPLAY_BUFFER_SIZE = 2000 # We prime the replay buffer with this many
								   # transition samples from a random agent.
	BATCH_SIZE = 100 # number of transitions to sample from replay buffer 
					 # for one optimization step, each timestep
	EPS = 0.5
	EPS_DECAY = 1-2e-4 # reach 0.1 after 23k timesteps
	MIN_EPS = 0.1


	### Display parameters 
	VERBOSE = False
	RENDER_PERIOD = 20  # Number of episodes between rendered episodes
	UPDATE_PLOT_PERIOD = 200 # Number of timesteps between updating the plot.
	LOSS_PLOT_MAX_POINTS = 100 # Number of points to use for the plot; When
							   # this exceeds the number of episodes we
							   # average the length of consecutive episodes
							   # to get approximately this many points total.


	q_net = CartPoleQNetwork(hidden_dims=HIDDEN_DIMS)

	target_net = CartPoleQNetwork(hidden_dims=HIDDEN_DIMS)
	target_net.set_weights(q_net.get_weights())

	replay_buffer = deque()
	episodes_since_last_render = 0
	render = False

	# Pre-load the buffer with experience from a random agent
	state = env.reset()
	for i in range(INIT_REPLAY_BUFFER_SIZE):
		action = env.action_space.sample()
		next_state, reward, done, _ = env.step(action) # 2
		replay_buffer.append((state, action, reward, next_state, done)) # 3
		if done:
			state = env.reset()
		else:
			state = next_state

	# Setup for training loop and plotting
	state = env.reset()
	t, i, longest_ep_duration = 0, 0, 0
	episode_lengths = []

	while True:
		"""
			Training loop:
				1. Choose action (epsilon-greedy)
				2. Environment step
				3. Record step in buffer - if done, record terminal state-action pairs,
					as we target these q-values to 0
				4. Optimize
					a. Sample random batch from buffer
					b. Compute targets for batch using target net
					c. Run optimization step on batch in Q-Net
				5. (Occasionally) Update target network weights from Q-Net

			We also update episode reward plot and optionally render at
			pre-set intervals.
		"""
		if render:
			env.render()

		## 1.
		action = q_net.get_action(state).numpy() if random.random() >= EPS \
			else env.action_space.sample()

		## 2.
		next_state, reward, done, _ = env.step(action) 

		## 3.
		# buffer entries are (s, a, r, ns, is_terminal_state)
		replay_buffer.append((state, action, reward, next_state, False)) 
		# if episode is over, add buffer entries for terminal state and each action
		if done: 
			for action in range(env.action_space.n):
				replay_buffer.append((next_state, action, 0, next_state, True))

		## 4.
		# a.
		batch = [random.choice(replay_buffer) for _ in range(BATCH_SIZE)] 
		s, a, r, ns, terminal = zip(*batch)
		s, a, r, ns, terminal = np.array(s, np.float32), np.array(a, np.int32), np.array(r, np.float32), np.array(ns, np.float32), np.array(terminal)
		# b.
		target = np.where(terminal, 0, r + DISCOUNT_RATE*target_net.value(ns).numpy())
		# c.
		loss = q_net.train_step(s, a, target)


		if VERBOSE:
			print(f"Iter: {i}, Q-Net batch loss: {loss}, Current episode duration: {t}, Longest episode so far: {longest_ep_duration}, eps: {EPS}")

		## 5.
		if i % UPDATE_TARGET_NET_PERIOD == 0:
			if VERBOSE:
				print("\n\nUPDATING TARGET NETWORK\n\n")
			target_net.set_weights(q_net.get_weights())

		# Plotting
		if i % UPDATE_PLOT_PERIOD == 0:
			group_size = int(round(len(episode_lengths)/LOSS_PLOT_MAX_POINTS))
			if group_size < 1:
				group_size = 1

			x = np.arange(0, len(episode_lengths), group_size)
			y = [np.mean(episode_lengths[i:i+group_size]) for i in range(0, len(episode_lengths), group_size)]

			plt.clf()
			plt.plot(x, y, c='blue')
			plt.title("Q-Net Agent Training Progress")
			plt.xlabel("Episodes experienced")
			plt.ylabel("Episode reward (moving average)")
			plt.draw()
			# plt.savefig(f"out_{}.png") TODO
			plt.pause(0.001)

		# Current episode is over
		if done: 
			episode_lengths.append(t)
			state, t = env.reset(), 0
			episodes_since_last_render += 1

			if render:
				render = False
				episodes_since_last_render = 0
			if not render and episodes_since_last_render == RENDER_PERIOD:
				render = True		
		# Current episode is not over
		else: 
			state, t = next_state, t+1
			longest_ep_duration = max(t, longest_ep_duration)
		i += 1

		EPS = max(EPS*EPS_DECAY, MIN_EPS)
	env.close()
