import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

def generate_p_matrix(num_states):
  p = np.zeros([num_states, num_states])
  for i in range(num_states):
    row_sum = 0
    for j in range(num_states):
      p[i][j] = np.random.rand(1)
      row_sum += p[i][j]
    p[i,:] = p[i,:]/row_sum
  return p

def generate_b_matrix(num_states, num_objects):
  b = np.zeros([num_states, num_objects])
  for i in range(num_states):
    row_sum = 0
    for j in range(num_objects):
      b[i][j] = np.random.rand(1)
      row_sum += b[i][j]
    b[i,:] = b[i,:]/row_sum
  return b

def next_state(p, cur_state):
  r = np.random.rand(1)
  state_transition_prob = p[cur_state-1]
  for i in range(len(state_transition_prob)):
    if r <= sum(state_transition_prob[:i+1]):
      return i+1 
  return len(state_transition_prob)

def current_observation(b, cur_state):
  r = np.random.rand(1)
  event_prob = b[cur_state-1]
  for i in range(len(event_prob)):
    if r <= sum(event_prob[:i+1]):
      return i+1
  return len(event_prob)

def forward(p, b, pi, obs):
  alpha = np.zeros((len(obs), p.shape[0]))
  alpha[0, :] = pi * b[:, V[0]-1]
  for t in range(1, len(obs)):
    for j in range(p.shape[0]):
      # Matrix Computation Steps
      #                  ((1x2) . (1x2))      *     (1)
      #                        (1)            *     (1)
      alpha[t, j] = alpha[t - 1].dot(p[:, j]) * b[j, obs[t]-1]

  return alpha

def viterbi(p, b, pi, obs):
  #decoding problem, using viterbi algo
  T = len(obs)
  N = p.shape[0]

  omega = np.zeros((T, N))
  omega[0, :] = np.log(pi * b[:, obs[0]-1])

  prev = np.zeros((T - 1, N))

  for t in range(1, T):
    for j in range(N):
      # Same as Forward Probability
      probability = omega[t - 1] + np.log(p[:, j]) + np.log(b[j, obs[t]-1])
      # This is our most probable state given previous state at time t (1)
      prev[t - 1, j] = np.argmax(probability)

      # This is the probability of the most probable state (2)
      omega[t, j] = np.max(probability)

  # Path Array
  S = np.zeros(T)

  # Find the most probable last hidden state
  last_state = np.argmax(omega[T - 1, :])

  S[0] = last_state

  backtrack_index = 1
  for i in range(T - 2, -1, -1):
    S[backtrack_index] = prev[i, int(last_state)]
    last_state = prev[i, int(last_state)]
    backtrack_index += 1

  # Flip the path array since we were backtracking
  S = np.flip(S, axis=0)

  # Convert numeric values to actual hidden states
  result = []
  for s in S:
    result.append(int(s+1))
  return result

"""Task 1"""

np.random.seed(200321513)
num_states = 4
num_objects = 3
S = (1,2,3,4) #states
V = (1,2,3) #set of all symbols
initial_distribution = np.array((1,0,0,0))

one_step_transition_matrix = generate_p_matrix(num_states)
print('Transition matrix')
print(one_step_transition_matrix)

print(f'Testing normalization for p matrix. Sum of p matrix rows: {np.sum(one_step_transition_matrix,axis=1)}')

event_matrix = generate_b_matrix(4,3)
print('Event matrix')
print(event_matrix)

print(f'Testing normalization for b matrix. Sum of b matrix rows: {np.sum(event_matrix, axis=1)}')

num_observations = 1000
observations = []
states = []
states.append(1) #initial state is 1

while len(observations) < num_observations:
  observations.append(current_observation(event_matrix, states[-1]))
  states.append(next_state(one_step_transition_matrix, states[-1]))

states = states[:-1]

"""Task 2"""

seq_obs = [1,2,3,3,1,2,3,3,1,2,3,3]
X = [[obs] for obs in LabelEncoder().fit_transform(observations)] 
X_test = [[obs] for obs in LabelEncoder().fit_transform(seq_obs)]
seq_prob = forward(one_step_transition_matrix, event_matrix, initial_distribution, seq_obs)
print(f'Probability that the sequence {seq_obs} came from the HMM: {sum(seq_prob[-1])}')

flag = False
for i in range(len(observations)-12):
  if seq_obs == observations[i:i+12]:
    flag = True
    print('The sequence exists in the generated observation')
if not flag:
  print('The sequence does not occur in the generated observation')

"""Task 3"""

print(f'Most probable state sequence for observations {seq_obs} is\n{viterbi(one_step_transition_matrix, event_matrix, initial_distribution, seq_obs)}')

"""Task 4"""

hmm_model = hmm.MultinomialHMM(n_components=num_states, random_state=10, n_iter=100)
hmm_model.fit(X)

transmission_matrix = hmm_model.transmat_
print('Transmision matrix')
print(transmission_matrix)

print(f'Testing for normalization on p matrix: {np.sum(transmission_matrix, axis=1)}')

emission_prob = hmm_model.emissionprob_
print('Emission probability')
print(emission_prob)

print(f'Testing for normalization on b matrix: {np.sum(emission_prob, axis=1)}')

initial_distribution = hmm_model.startprob_
print('Initial distribution')
print(initial_distribution)

print(f'Testing for normalization on initialization matrix: {np.sum(transmission_matrix, axis=1)}')

