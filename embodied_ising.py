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

	def get_state(self,mode='all'):
		if mode=='all':
			return self.s
		elif mode=='agent':
			return self.s[0:self.Asize]
		elif mode=='environment':
			return self.s[self.Asize:]
	
	def get_agent_state_index(self,mode='all'):
		if mode=='all':
			return bool2int(0.5*(self.s+1))
		elif mode=='agent':
			return bool2int(0.5*(self.s[0:self.Asize]+1))
		elif mode=='environment':
			return bool2int(0.5*(self.s[self.Asize:]+1))
			
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
					self.J[i,j]=np.random.rand(1)*std
		
	#Execute step of the Glauber algorithm to update the state of one unit
	def GlauberStep(self,i=None):			
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
			
	#Execute step of the Glauber algorithm to update the state of one unit restricting its influences to a given range of units	
	def RestrictedGlauberStep(self,i,rng,bias=1):			#Execute step of Glauber algorithm
		eDiff = self.deltaErng(i,rng,bias)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
	
	
	#Compute energy difference between two states with a flip of spin i	
	def deltaE(self,i):		
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
 
 	#Compute energy difference between two states with a flip of spin i, considering only a restricted range of connections
	def deltaErng(self,i,rng,bias=1):		
		return 2*(self.s[i]*self.h[i]*bias + np.sum(self.s[i]*(self.J[i,rng]*self.s[rng])+self.s[i]*(self.J[rng,i]*self.s[rng])))
			
	
	#Update states of the agent, where sensors are only influenced by units in the environment
	def UpdateAgent(self):			
		inds=np.random.permutation(self.Asize)
		rngS=np.arange(self.Asize,self.size)
#		rngS=np.concatenate((self.Spos,np.arange(self.Asize,self.size)))
		rngA=np.arange(self.Asize)
		for i in self.Spos:
			self.RestrictedGlauberStep(i,rngS,bias=0)
		for i in inds:
			if not i in self.Spos:
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
	def ContrastiveDivergence(self,Iterations,T=None):	
		if T==None:
			T=self.defaultT
		u=0.04
		count=0
		self.observables_positive()
		self.observables_negative()
		fit = KL(self.PVpos,self.PVneg)
		print(self.size,count,fit)
		# Main simulation loop:
		for t in range(Iterations):
			count+=1
			dh=u*(self.mpos-self.mneg)
			dJ=u*(self.Cpos-self.Cneg)
			
			self.h[0:self.Asize]+=dh[0:self.Asize]
			self.J[0:self.Asize,0:self.Asize]+=dJ[0:self.Asize,0:self.Asize]
			
			self.observables_positive(T)
			self.observables_negative(T)
			fit = KL(self.PVpos,self.PVneg)
			if count%10==0:
				print(self.size,count,fit)
			
			
			
	#Critical Learning Algorithm for poising the system in a critical state	
	def CriticalGradient(self,T):
	
		dh=np.zeros((self.size))
		dJ=np.zeros((self.size,self.size))
	
		E=0
		E2=0
		
		Esm=np.zeros(self.size)
		E2sm=np.zeros(self.size)
		m=np.zeros(self.size)
		
		EsC=np.zeros((self.size,self.size))
		E2sC=np.zeros((self.size,self.size))
		C=np.zeros((self.size,self.size))
		
		self.randomize_state()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.UpdateAgent()
			self.UpdateEnvironment()
			n=bool2int((self.s+1)/2)
			Es=-(np.dot(self.s,self.h) + np.dot(np.dot(self.s,self.J),self.s))
			E+=Es/T
			E2+=Es**2/T
			for i in range(self.size):
				m[i]+=self.s[i]/float(T)
				Esm[i]+=Es*self.s[i]/float(T)
				E2sm[i]+=Es**2*self.s[i]/float(T)
				for j in np.arange(i+1,self.size):
					C[i,j]+=self.s[i]*self.s[j]/float(T)
					EsC[i,j]+=Es*self.s[i]*self.s[j]/float(T)
					E2sC[i,j]+=Es**2*self.s[i]*self.s[j]/float(T)
		
		dh=m*(2*E+2*E**2-E2)-2*Esm*(1+E)+E2sm
		dJ=C*(2*E+2*E**2-E2)-2*EsC*(1+E)+E2sC
		
		self.HC=(E2-E**2)
		
		return dh,dJ
		
	#Dynamical Critical Learning Algorithm for poising units in a critical state
	def DynamicalCriticalGradient(self,T=None):
		if T==None:
			T=self.defaultT
		dh=np.zeros((self.size))
		dJ=np.zeros((self.size,self.size))
		
		msH=np.zeros(self.size)
		mF=np.zeros(self.size)
		mG=np.zeros(self.size)
				
		msh=np.zeros(self.size)
		msFh=np.zeros(self.size)
		msGh=np.zeros(self.size)
		mdFh=np.zeros(self.size)
		mdGh=np.zeros(self.size)
		ms2Hh=np.zeros(self.size)
		
		msJ=np.zeros((self.size,self.size))
		msFJ=np.zeros((self.size,self.size))
		msGJ=np.zeros((self.size,self.size))
		mdFJ=np.zeros((self.size,self.size))
		mdGJ=np.zeros((self.size,self.size))
		ms2HJ=np.zeros((self.size,self.size))
		
		self.randomize_state()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.UpdateAgent()
			self.UpdateEnvironment()
			H= self.h + np.dot(self.s,self.J)+ np.dot(self.J,self.s)
			F = H*np.tanh(H)-np.log(2*np.cosh(H))
			G = (H/np.cosh(H))**2 + self.s*H*F
			dF = H/np.cosh(H)**2
			dG = 2*H*(1-H*np.tanh(H))/np.cosh(H)**2 + self.s*F + self.s*H*dF
			
			msH+=self.s*H/float(T)
			mF+=F/float(T)
			mG+=G/float(T)
			
			
			msh+=self.s/float(T)
			msFh+=self.s*F/float(T)
			msGh+=self.s*G/float(T)
			mdFh+=dF/float(T)
			mdGh+=dG/float(T)
			ms2Hh+=H/float(T)
			
			for j in range(self.size):
				msJ[j,:]+=self.s*self.s[j]/float(T)
				msFJ[j,:]+=self.s*self.s[j]*F/float(T)
				msGJ[j,:]+=self.s*self.s[j]*G/float(T)
				mdFJ[j,:]+=self.s[j]*dF/float(T)
				mdGJ[j,:]+=self.s[j]*dG/float(T)
				ms2HJ[j,:]+=self.s[j]*H/float(T)
			
		dh = mdGh + msGh - msh*mG - (msh+ms2Hh-msh*msH)*mF - msH*(mdFh+msFh-msh*mF)
		dJ1 = mdGJ + msGJ - msJ*mG - (msJ+ms2HJ-msJ*msH)*mF - msH*(mdFJ+msFJ-msJ*mF)
		

		dJ=np.zeros((self.size,self.size))
		for j in range(self.size):
			for i in np.arange(self.size):
				if i>j:
					dJ[j,i]+=dJ1[j,i]
				elif j>i:
					dJ[i,j]+=dJ1[j,i]
		
		self.HCl=mG-msH*mF
		self.HC=np.sum(self.HCl[0:self.Asize])
		
		return dh,dJ
		
		
	def CriticalLearning(self,Iterations,T=None,mode='dynamic'):	
		u=0.04
		count=0
		if mode=='static':
			dh,dJ=self.CriticalGradient(T)
		elif mode=='dynamic':
			dh,dJ=self.DynamicalCriticalGradient(T)
		fit=self.HC
		print(self.size,count,fit)
		for i in range(Iterations):
			count+=1
			self.h[0:self.Asize]+=u*dh[0:self.Asize]
			self.J[0:self.Asize,0:self.Asize]+=u*dJ[0:self.Asize,0:self.Asize]
			if mode=='static':
				dh,dJ=self.CriticalGradient(T)
			elif mode=='dynamic':
				dh,dJ=self.DynamicalCriticalGradient(T)
			fit=self.HC
			
			if count%10==0:
				print(self.size,count,fit)
			

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

	
