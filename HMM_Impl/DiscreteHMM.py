import numpy
import random

class DiscreteHMM:
	def __init__(self, n_states, n_obs):
		self.n_states = n_states
		self.random_state = numpy.random.RandomState(0)
		
		#Normalize random initial state
		self.pi = self._normalize(self.random_state.rand(1, self.n_states))
		self.A = self._normalize2(self.random_state.rand(self.n_states, self.n_states))

		self.b = None
		self.n_obs = n_obs
		
	def _viterbi(self, B):
		likelihood = 0.0
		T = B.shape[0]
		delta = numpy.zeros(B.shape)
		prob_seq = []
		seq = []
		
		for t in range(T):
			if t == 0:
				delta[t,:] = B[t,:] * self.pi.ravel()
			else:
				prob_terms = (delta[t-1,:] * (self.A * B[t,:]).T).T
				delta[t,:] = numpy.max(prob_terms, axis=0)
				i = numpy.argmax(prob_terms, axis=0)
				prob_seq.append(i)
				
		last_state = numpy.argmax(delta[T-1,:])
		seq.append(last_state)
		for t in range(T-2, -1, -1):
			last_state = prob_seq[t][last_state]
			seq.append(last_state)
			
		likelihood = delta[T-1,seq[0]]
			
		return numpy.array(seq[::-1]), likelihood

	def _forward(self, B):
		likelihood = 0.0
		T = B.shape[0]
		alpha = numpy.zeros(B.shape)
		for t in range(T):
			if t == 0:
				alpha[t, :] = B[t, :] * self.pi.ravel()
			else:
				alpha[t, :] = B[t, :] * numpy.dot(self.A.T, alpha[t - 1, :])
			alpha_sum = numpy.sum(alpha[t, :])
			#alpha[t, :] /= alpha_sum
		likelihood = numpy.log(numpy.sum(alpha[T-1,:]))
		return likelihood, alpha

	def _backward(self, B):
		T = B.shape[0]
		beta = numpy.zeros(B.shape);
        
		beta[-1,:] = numpy.ones(B.shape[1])
            
		for t in range(T - 2, -1, -1):
			beta[t,:] = numpy.dot(self.A, (B[t + 1,:] * beta[t + 1,:]))
			#beta[t,:] /= numpy.sum(beta[t,:])
		return beta
    
	def _state_likelihood(self, obs):
		B = numpy.zeros((obs.shape[0], self.n_states))
		for s in range(self.n_states):
			B[:, s] = numpy.array([self.b[s,i-1] for i in obs[:,0]])
		return B

	def _normalize(self, x):
		return x / numpy.sum(x)
    
	def _normalize2(self, x):
		return x / numpy.sum(x, axis=0)
    
	def _bw_init(self, obs):
		if self.b is None:
			self.b = numpy.zeros((self.n_states, self.n_obs))
			self.b[:,:] = [[1.0/self.n_obs for i in range(self.n_obs)] for j in range(self.n_states)] 
		return self
    
	def _baum_welch(self, obs): 
		obs = numpy.atleast_2d(obs)
		B = self._state_likelihood(obs)
		T = obs.shape[0]
        
		likelihood, alpha = self._forward(B)
		beta = self._backward(B)
        
		gamma = numpy.zeros((T, self.n_states))
		csi = numpy.zeros((T-1, self.n_states, self.n_states))
        
		for t in range(T - 1):		
			alpha_beta = alpha[t,:] * beta[t,:]
			gamma[t,:] = self._normalize(alpha_beta)
			partial_csi = ((self.A*(beta[t+1, :] * B[t+1, :])).T * alpha[t,:]).T
			csi[t,:,:] = self._normalize(partial_csi)
			
              
		alpha_beta = alpha[-1,:] * beta[-1,:]
		gamma[-1,:] = self._normalize(alpha_beta)
        
		expected_pi = gamma[0,:]
		expected_A = (numpy.sum(csi, axis=0) / numpy.sum(gamma[1:,:], axis=0)).T
		
		expected_b = numpy.zeros((self.n_states, self.n_obs))
        
		for s in range(self.n_states):
			for k in range(self.n_obs):
				for t in range(T):
					expected_b[s,k] += (obs[t,0] == k+1)*gamma[t,s]
				expected_b[s,k] /= numpy.sum(gamma[:,s])
				
        
		self.pi = expected_pi
		self.b = expected_b
		self.A = expected_A
		return likelihood
    
	def _roll(self, bias):
		number = random.uniform(0, sum(bias))
		current = 0
		for i, bias in enumerate(bias):
			current += bias
			if number <= current:
				return i + 1
	
	def fit(self, obs, n_iter=15):
		#Obs should be n_examples, n_dim
		obs = numpy.atleast_2d(obs)
		if len(obs.shape) == 2:
			for i in range(n_iter):
				self._bw_init(obs)
				log_likelihood = self._baum_welch(obs)
		return self
    
	def seq_prob(self, obs):
		#Obs should be n_examples, n_dim
		obs = numpy.atleast_2d(obs)
		if len(obs.shape) == 2:
			B = self._state_likelihood(obs)
			likelihood, _ = self._forward(B)
			return likelihood
			
	def state_seq(self, obs):
		#Obs should be n_examples, n_dim
		obs = numpy.atleast_2d(obs)
		if len(obs.shape) == 2:
			B = self._state_likelihood(obs)
			seq, likelihood = self._viterbi(B)
			return seq, likelihood
			
	def emit_sequence(self, size):
		emit_seq = numpy.zeros((size,1))
		state = None
		for i in range(size):
			if i == 0:
				state = self._roll(self.pi)-1
			else:
				state = self._roll(self.A[state,:])-1
			emit_seq[i,0] = self._roll(self.b[state,:])
		return emit_seq
			
def roll(bias):
	number = random.uniform(0, sum(bias))
	current = 0
	for i, bias in enumerate(bias):
		current += bias
		if number <= current:
			return i + 1
	
def write_output(name, model, train):
	#train: (obs_train, train_states, train_state_likelihood)	
	obs_train, train_states, train_state_likelihood = train
	
	with file(name, 'w') as output:
			output.write('HMM WITH MULTIVARIATE DISCRETE EMISSION DISTRIBUTION\n\n')
			output.write('==MODEL PARAMETERS==\n')
			output.write('# states: {0}\n'.format(model.n_states))
			
			output.write('\n# pi shape: {0}\n'.format(model.pi.shape))
			numpy.savetxt(output, model.pi, fmt='%-7.7f')
			
			output.write('\n# transition A shape: {0}\n'.format(model.A.shape))
			numpy.savetxt(output, model.A, fmt='%-7.2f')
			
			output.write('\n# b shape: {0}\n'.format(model.b.shape))
			numpy.savetxt(output, model.b, fmt='%-7.2f')
			
			output.write('\n\n==RESULTS==\n')
			output.write('# Obs set shape: {0}\n'.format(obs_train.shape))
			output.write('# Obs set: {0}\n'.format(obs_train[:,0]))
			output.write('State sequence: {0}\n'.format(train_states))
			output.write('State sequence likelihood: {0}\n'.format(train_state_likelihood))


if __name__ == "__main__":
	sum_dice = numpy.sum(range(6))
	dice1 = [1.0/6.0 for i in range(6)]
	dice2 = [(i+1.0)/sum_dice for i in range(6)]
	
	model = DiscreteHMM(2, 6)
	model.b = numpy.zeros((2, 6))
	model.b[0,:] = numpy.array(dice1)
	model.b[1,:] = numpy.array(dice2)
	model.A = numpy.array([[0.5,0.5],[0.5,0.5]])
	model.pi = numpy.array([0.5,0.5])
	obs_train = model.emit_sequence(10)
	
	train_states, train_state_likelihood = model.state_seq(obs_train)
	
	train = (obs_train, train_states, train_state_likelihood)

	write_output('DiscreteHMM.out', model, train)
	
	model = DiscreteHMM(2, 6)
	model.fit(obs_train)
	
	train_states, train_state_likelihood = model.state_seq(obs_train)
	
	train = (obs_train, train_states, train_state_likelihood)
	write_output('DiscreteHMM2.out', model, train)
	