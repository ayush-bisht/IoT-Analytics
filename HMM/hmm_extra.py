import numpy as np
import matplotlib.pyplot as plt
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

def plot(y, y_label):
  fig = plt.figure(figsize= (7,6))
  x = list(range(2,20))
  plt.plot(x, y, marker='o')
  plt.xlabel('Number of states')
  plt.ylabel(y_label)
  plt.title(y_label+' VS Number of states')
  fig.patch.set_facecolor('w')
  plt.show()

def current_observation(b, cur_state):
  r = np.random.rand(1)
  event_prob = b[cur_state-1]
  for i in range(len(event_prob)):
    if r <= sum(event_prob[:i+1]):
      return i+1
  return len(event_prob)

np.random.seed(200321513)
num_objects = 3
num_states = 4
V = (1,2,3) #set of all symbols
pi = [1,0,0,0]

one_step_transition_matrix = generate_p_matrix(num_states)

event_matrix = generate_b_matrix(4,3)

print(f'Testing for normalization, sum of b matrix rows: {np.sum(event_matrix, axis=1)}')

num_observations = 5000
observations = []
states = []
states.append(1) #initial state is 1

while len(observations) < num_observations:
  observations.append(current_observation(event_matrix, states[-1]))
  states.append(next_state(one_step_transition_matrix, states[-1]))

states = states[:-1]

aic_list = []
bic_list = []
log_likelihood_list = []
X = [[obs] for obs in LabelEncoder().fit_transform(observations)]

for n in range(2, 20):
  hmm_model = hmm.MultinomialHMM(n_components=n, random_state=10)
  hmm_model.fit(X)
  log_likelihood = hmm_model.score(X)
  num_parameters = n*n + n*num_objects + n
  aic = -2*log_likelihood + 2*num_parameters
  bic = -2*log_likelihood + num_parameters* np.log(len(observations))
  log_likelihood_list.append(log_likelihood)
  aic_list.append(aic)     
  bic_list.append(bic)

plot(log_likelihood_list, 'Log likelihood')

plot(aic_list, 'AIC')

plot(bic_list, 'BIC')

#final model
hmm_model = hmm.MultinomialHMM(n_components=4, random_state=10)
hmm_model.fit(X)

print('Transmission probability')
print(hmm_model.transmat_)

print('Emission probability')
print(hmm_model.emissionprob_)

print('Initial probability')
print(hmm_model.startprob_)

