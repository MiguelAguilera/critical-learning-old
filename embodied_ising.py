import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class ising:
	#Initialize the network
	def __init__(self, agentsize,envsize=0,sensorunits=np.array([]),motorunits=np.array([]) ):	#Create ising model
	
		netsize=agentsize+envsize
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
		self.Beta=1.0
		self.defaultT=netsize*100
		
		self.Spos=sensorunits				#List of sensor units
		self.Mpos=motorunits				#List of motor units
		self.Asize=agentsize			#Number of units in the agent
		self.Envsize=envsize			#Number of units in the environment		
		self.Ssize=len(sensorunits)		#Number of sensor units
		self.Msize=len(motorunits)		#Number of motor units							

	#Randomize the state of (part of) the network
	def randomize_state(self,mode=None):
		if mode is None:
			self.s = np.random.randint(0,2,self.size)*2-1
		elif mode=='agent':
			self.s[0:self.Asize] = np.random.randint(0,2,self.Asize)*2-1
		elif mode=='environment':
			self.s[self.Asize:self.size] = np.random.randint(0,2,self.size-self.Asize)*2-1
		elif mode=='sensors':
			self.s[self.S] = np.random.randint(0,2,self.Ssize)*2-1
		elif mode=='motors':
			self.s[self.M] = np.random.randint(0,2,self.Msize)*2-1

	#Set random bias to sets of units of the system
	def random_fields(self,mode='all',std=1):
		if mode=='all':
			inds=np.arange(self.size)
		elif mode=='agent':
			inds=np.arange(self.Asize)
		elif mode=='environment':
			inds=np.arange(self.Asize,self.size)		
		self.h[inds]=np.random.randn(len(inds))*std		
		
	#Set random connections to sets of units of the system
	def random_wiring(self,mode='environment',std=1):	#Set random values for h and J
		if mode == 'agent':
			inds1=np.arange(0,self.Asize)
			inds2=np.arange(0,self.Asize)
		elif mode=='environment':
			inds1=np.arange(self.Asize,self.size)
			inds2=np.arange(self.Asize,self.size)
		elif mode=='sensors':
			inds1=self.Spos
			inds2=np.arange(self.Asize,self.size)
		elif mode=='motors':
			inds1=self.Mpos
			inds2=np.arange(self.Asize,self.size)
		for i in inds1:
			for j in inds2:
				if i<j:
					self.J[i,j]=np.random.randn(1)*std
		
	#Execute step of the Glauber algorithm to update the state of one unit
	def GlauberStep(self,i=None):			
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
			
	#Execute step of the Glauber algorithm to update the state of one unit restricting its influences to a given range of units	
	def RestrictedGlauberStep(self,i,rng):			#Execute step of Glauber algorithm
		eDiff = self.deltaErng(i,rng)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
	
	
	#Compute energy difference between two states with a flip of spin i	
	def deltaE(self,i):		
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
 
 	#Compute energy difference between two states with a flip of spin i, considering only a restricted range of connections
	def deltaErng(self,i,rng):		
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,rng]*self.s[rng])+self.s[i]*(self.J[rng,i]*self.s[rng])))
			
	
	#Update states of the agent, where sensors are only influenced by units in the environment
	def UpdateAgent(self):			
		inds=np.random.permutation(self.Asize)
		rngS=np.arange(self.Asize,self.size)
		rngA=np.arange(self.Asize)
		for i in inds:
			if i in self.Spos:
				self.RestrictedGlauberStep(i,rngS)
			else:
				self.RestrictedGlauberStep(i,rngA)
	
	#Update states of the 'dreaming' agent, where sensors are influenced by the agent's units
	def UpdateDreamingAgent(self):	
		inds=np.random.permutation(self.Asize)
		rng=np.arange(self.Asize)
		for i in inds:
			self.RestrictedGlauberStep(i,rng)
			
	#Update states of the environment, where only motor units from the agent infuence the environment		
	def UpdateEnvironment(self):	
		inds=np.random.permutation(self.Envsize)+self.Asize
		rng=np.arange(self.Asize,self.size)
		rng =  np.concatenate((self.Mpos,rng))
		for i in inds:
			self.RestrictedGlauberStep(i,rng)	

	#Update all states of the system without restricted infuences
	def SequentialGlauberStep(self):	
		inds=np.random.permutation(self.size)
		for i in inds:
			self.GlauberStep(i)

	#Get mean and correlations from simuations of the positive ('embodied') phase
	def observables_positive(self,T=None):		
		if T==None:
			T=self.defaultT
		self.mpos=np.zeros((self.size))
		self.Cpos=np.zeros((self.size,self.size))
		self.PVpos=np.ones(2**self.Ssize)
			
		self.randomize_state()
		for t in range(T):
			self.UpdateAgent()
			self.UpdateEnvironment()
			
			n=int(bool2int(0.5*(self.s[self.Spos]+1)))
			self.PVpos[n]+=1.0
			
			self.mpos+=self.s/float(T)
			for i in range(self.size):
				self.Cpos[i,i+1:]+=self.s[i]*self.s[i+1:]/float(T)
					
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.Cpos[i,j]-=self.mpos[i]*self.mpos[j]	
		self.PVpos/=float(np.sum(self.PVpos))
				
	
	#Get mean and correlations from simuations of the negative ('dreaming') phase
	def observables_negative(self,T=None):		
		if T==None:
			T=self.defaultT
		self.mneg=np.zeros((self.size))
		self.Cneg=np.zeros((self.size,self.size))
		self.PVneg=np.ones(2**self.Ssize)
				
		self.randomize_state()
		for t in range(T):
			self.UpdateDreamingAgent()
			
			n=int(bool2int(0.5*(self.s[self.Spos]+1)))
			self.PVneg[n]+=1.0
			
			self.mneg+=self.s/float(T)
			for i in range(self.size):
				self.Cneg[i,i+1:]+=self.s[i]*self.s[i+1:]/float(T)
					
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.Cneg[i,j]-=self.mneg[i]*self.mneg[j]	
		self.PVneg/=float(np.sum(self.PVneg))
	
	#Contrastive Divergence Algorithm for learning the probability distribution of sensor units
	def ContrastiveDivergence(self,T):	

		u=0.04
		count=0
		self.observables_positive()
		self.observables_negative()
		# Main simulation loop:
		for t in range(T):

			dh=u*(self.mpos-self.mneg)
			self.h[0:self.Asize]+=dh[0:self.Asize]
			dJ=u*(self.Cpos-self.Cneg)
			self.J[0:self.Asize,0:self.Asize]+=dJ[0:self.Asize,0:self.Asize]
			
			self.observables_positive()
			self.observables_negative()
#			fit = max (np.max(np.abs(self.mpos[0:self.Asize]-self.mneg[0:self.Asize])),np.max(np.abs(self.Cpos[0:self.Asize,0:self.Asize]-self.Cneg[0:self.Asize,0:self.Asize])))
			fit = KL(self.PVpos,self.PVneg)
			if count%10==0:
				print(self.size,count,fit)
			count+=1

#Transform bool array into positive integer
def bool2int(x):				
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
        y += j*2**i
    return y
    
#Transform positive integer into bit array
def bitfield(n,size):			
    x = [int(x) for x in bin(n)[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)

#Extract subset of a probability distribution
def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
#Compute Entropy of a distribution
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log(P[n])
	return E
	
#Compute Mutual Information between two distributions
def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
#Compute TSE complexity of a distribution
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
#Compute the Kullback-Leibler divergence between two distributions
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
 
#Compute the Jensen-Shannon divergence between two distributions   
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))

	
