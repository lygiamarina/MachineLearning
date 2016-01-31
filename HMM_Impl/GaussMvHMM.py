import scipy.stats
import numpy

class GaussMvHMM:
	def __init__(self, n_states):
		self.n_states = n_states
		self.random_state = numpy.random.RandomState(0)
		
		#Normalize random initial state
		self.pi = self._normalize(self.random_state.rand(1, self.n_states))
		self.A = self._normalize2(self.random_state.rand(self.n_states, self.n_states))
		
		self.mu = None
		self.cov = None
		self.n_dim = None
		
	def _viterbi(self, B):
		likelihood = 0.0
		T = B.shape[0]
		delta = numpy.zeros(B.shape)
		prob_seq = []
		seq = []
		
		for t in range(T):
			if t == 0:
				delta[t,:] = B[t,:] * self.pi
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
		log_likelihood = 0.0
		T = B.shape[0]
		alpha = numpy.zeros(B.shape)
		for t in range(T):
			if t == 0:
				alpha[t, :] = B[t, :] * self.pi.ravel()
			else:
				alpha[t, :] = B[t, :] * numpy.dot(self.A.T, alpha[t - 1, :])
         
			alpha_sum = numpy.sum(alpha[t, :])
			alpha[t,:] /= alpha_sum
			log_likelihood = log_likelihood + numpy.log(alpha_sum)
		return log_likelihood, alpha

	def _backward(self, B):
		T = B.shape[0]
		beta = numpy.zeros(B.shape);
        
		beta[-1,:] = numpy.ones(B.shape[1])
            
		for t in range(T - 2, -1, -1):
			beta[t,:] = numpy.dot(self.A, (B[t + 1,:] * beta[t + 1,:]))
			beta[t,:] /= numpy.sum(beta[t,:])
		return beta
    
	def _state_likelihood(self, obs):
		obs = numpy.atleast_2d(obs)
		B = numpy.zeros((obs.shape[0], self.n_states))
		for s in range(self.n_states):
			#scipy 0.14
			B[:, s] = scipy.stats.multivariate_normal.pdf(obs, mean=self.mu[s, :].T, cov=self.cov[s, :, :].T)
		return B

	def _normalize(self, x):
		return x / numpy.sum(x)
    
	def _normalize2(self, x):
		return x / numpy.sum(x, axis=0)
    
	def _bw_init(self, obs):
		if self.n_dim is None:
			self.n_dim = obs.shape[1]
		if self.mu is None:
			subset = self.random_state.choice(numpy.arange(self.n_dim), size=self.n_states, replace=False)
			self.mu = obs[subset, :]
		if self.cov is None:
			self.cov = numpy.zeros((self.n_states, self.n_dim, self.n_dim))
			self.cov += numpy.diag(numpy.diag(numpy.cov(obs, rowvar=0)))
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
        
		expected_mu = numpy.zeros((self.n_states, self.n_dim))
		expected_cov = numpy.zeros((self.n_states, self.n_dim, self.n_dim))
        
		gamma_state_sum = numpy.sum(gamma, axis=0)
		gamma_state_sum = gamma_state_sum + 0.0001*(gamma_state_sum == 0)
        
		for s in range(self.n_states):
			gamma_obs = obs.T * gamma[:,s]
			expected_mu[s, :] = numpy.sum(gamma_obs, axis=1) / gamma_state_sum[s]
			
			partial_cov = numpy.zeros((self.n_dim, self.n_dim))
			for t in range(T):
				obs_mu = obs[t,:]-expected_mu[s,:]
				partial_cov += gamma[t,s]*numpy.dot(obs_mu[numpy.newaxis].T, obs_mu[numpy.newaxis])
			expected_cov[s,:,:] = partial_cov / gamma_state_sum[s]
        
		#Positive semidefinite
		expected_cov += 0.0001 * numpy.eye(self.n_dim)
        
		self.pi = expected_pi
		self.mu = expected_mu
		self.cov = expected_cov
		self.A = expected_A
		return likelihood
    
	def _roll(self, bias):
		number = random.uniform(0, sum(bias))
		current = 0
		for i, bias in enumerate(bias):
			current += bias
			if number <= current:
				return i
	
	def fit(self, obs, n_iter=15):
		#Obs should be n_examples, n_dim
		if len(obs.shape) == 2:
			for i in range(n_iter):
				self._bw_init(obs)
				log_likelihood = self._baum_welch(obs)
		return self
    
	def seq_prob(self, obs):
		#Obs should be n_examples, n_dim
		if len(obs.shape) == 2:
			B = self._state_likelihood(obs)
			likelihood, _ = self._forward(B)
			return likelihood
			
	def state_seq(self, obs):
		#Obs should be n_examples, n_dim
		if len(obs.shape) == 2:
			B = self._state_likelihood(obs)
			seq, likelihood = self._viterbi(B)
			return seq, likelihood
			
	def emit_sequence(self, size):
		emit_seq = numpy.zeros((size,self.n_dim))
		state = None
		for i in range(size):
			if i == 0:
				state = self._roll(self.pi[0,:])
			else:
				state = self._roll(self.A[state,:])
			emit_seq[i,:] = numpy.random.multivariate_normal(mu[state,:], cov[state,:])
		return emit_seq
	