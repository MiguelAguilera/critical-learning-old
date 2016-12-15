import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class ising:
	#Initialize the network
	def __init__(self, netsize,Nsensors=1,envsize=20,envstd=3):	#Create ising model
	
		self.size=netsize
		self.Ssize=Nsensors			#Number of sensors
		self.envsize=envsize		#Environment size
		
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
		self.randomize_position()
		
		self.Beta=1.0
		self.defaultT=max(100,netsize*20)
		
		x=np.arange(envsize)
		self.mu=envsize*0.5
		self.sig=envstd
		self.maxgradient=np.max(np.diff(self.Sense(x)))*1.01
		self.mingradient=np.min(np.diff(self.Sense(x)))*1.01
		self.sensorbins=np.linspace(self.mingradient,self.maxgradient,2**self.Ssize+1)
#		self.env=np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
		self.Update(0)
		


	def get_state(self,mode='all'):
		if mode=='all':
			return self.s
		elif mode=='motors':
			return self.s[-1]
		elif mode=='sensors':
			return self.s[0:self.Ssize]
	
	def get_agent_state_index(self,mode='all'):
		return bool2int(0.5*(self.get_state(mode)+1))
			
	#Randomize the state of the network
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1

	def randomize_position(self):
		self.pos=np.random.randint(self.envsize)
	#Set random bias to sets of units of the system
	def random_fields(self,std=1):
		self.h[self.Ssize:]=np.random.randn(self.size-self.Ssize)*std		
		
	#Set random connections to sets of units of the system
	def random_wiring(self,std=1,offset=0):	#Set random values for h and J
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				if i<j and (i>=self.Ssize or j>=self.Ssize):
					self.J[i,j]=(np.random.randn(1)+offset)*std
		
	def Sense(self,x):
		return np.exp(-np.power(x - self.mu, 2.) / (2 * np.power(self.sig, 2.)))


	def Move(self):
		past_position=self.pos
		self.pos=(self.pos+self.s[-1])% self.envsize	
		self.sensor=self.Sense(self.pos)-self.Sense(past_position)
		
	def UpdateSensors(self):
#		self.s[0:self.Ssize]= 2*bitfield(np.floor(self.sensor*2**self.Ssize),self.Ssize)-1
		self.s[0:self.Ssize]= 2*bitfield(np.digitize(self.sensor,self.sensorbins)-1,self.Ssize)-1
		
	#Execute step of the Glauber algorithm to update the state of one unit
	def GlauberStep(self,i=None):			
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
			
	#Compute energy difference between two states with a flip of spin i	
	def deltaE(self,i):		
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
		
		
	#Update states of the agent from its sensors		
	def Update(self,i=None):	
		if i is None:
			i = np.random.randint(self.size)
		if i==0:
			self.Move()
			self.UpdateSensors()
		elif i>=self.Ssize:
			self.GlauberStep(i)
	
	#Update states of the agent in dreaming mode			
	def UpdateDreaming(self,i=None):	
		if i is None:
			i = np.random.randint(self.size)
		self.GlauberStep(i)
			
	def SequentialUpdate(self):
		for i in np.random.permutation(self.size):
			self.Update(i)

	def SequentialUpdateDreaming(self):
		for i in np.random.permutation(self.size):
			self.UpdateDreaming(i)
	
	#Update all states of the system without restricted infuences
	def SequentialGlauberStep(self):	
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)

	#Get mean and correlations from simuations of the positive ('embodied') phase
	def observables_positive(self,T=None):		
		if T==None:
			T=self.defaultT
		self.mpos=np.zeros((self.size))
		self.Cpos=np.zeros((self.size,self.size))
		self.PVpos=np.ones(2**self.Ssize)
			
		self.randomize_state()
		self.randomize_position()
		for t in range(T):
			self.SequentialUpdate()
			
			n=int(bool2int(0.5*(self.s[0:self.Ssize]+1)))
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
		self.randomize_position()
		for t in range(T):
			self.SequentialUpdateDreaming()
			
			n=int(bool2int(0.5*(self.s[0:self.Ssize]+1)))
			self.PVneg[n]+=1.0
			
			self.mneg+=self.s/float(T)
			for i in range(self.size):
				self.Cneg[i,i+1:]+=self.s[i]*self.s[i+1:]/float(T)
					
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.Cneg[i,j]-=self.mneg[i]*self.mneg[j]	
		self.PVneg/=float(np.sum(self.PVneg))
	
	#Boltzmann learning Algorithm for learning the probability distribution of sensor units
	def BoltzmannLearning(self,Iterations,T=None):	
		if T==None:
			T=self.defaultT
		u=0.04
		count=0
		self.observables_positive()
		self.observables_negative()
		fit = KL(self.PVpos,self.PVneg)
		print(count,fit)
		# Main simulation loop:
		for t in range(Iterations):
			count+=1
			dh=u*(self.mpos-self.mneg)
			dJ=u*(self.Cpos-self.Cneg)
			
#			dh[self.Spos]=0
#			dJ[self.Spos[:,None],self.Spos]=0
			self.h+=dh
			self.J+=dJ
			
			self.observables_positive(T)
			self.observables_negative(T)
			fit = KL(self.PVpos,self.PVneg)
			if count%10==0:
				print(count,fit)
			
			
			
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
		self.randomize_position()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			for i in range(self.size):
				self.Update()
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
		
		self.HC=(E2-E**2)/float(self.size)
		
		return dh,dJ
		
	#Dynamical Critical Learning Algorithm for poising units in a critical state
	def DynamicalCriticalGradient(self,T=None):
		if T==None:
			T=self.defaultT
		
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
		self.randomize_position()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.SequentialUpdate()
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
		
		dh[0:self.Ssize]=0
		for j in range(self.size):
			for i in range(self.size):
				if i in range(self.Ssize):
					dJ1[j,i]=0				#Remove wrong-way change in unidirectional couplings
					dJ1[i,j]+=dJ1[i,j]		#Multiply by two change in unidirectional couplings
				if i==j:
					dJ1[i,i]=0

		for j in range(self.size):
			for i in range(self.size):
				if i>j:
					dJ[j,i]+=dJ1[j,i]
				elif j>i:
					dJ[i,j]+=dJ1[j,i]
		
		self.HCl=mG-msH*mF
		self.HC=np.sum(self.HCl[self.Ssize:])
		
#		dh[0:self.Ssize]=0
##		dJ[0:self.Ssize,0:self.Ssize]=0
#		for i in range(self.Ssize):
#			self.h[i]=-0.5*np.log((1-msh[i])/(1+msh[i]))
		
		return dh,dJ
		
		
	def CriticalLearning(self,Iterations,T=None,mode='dynamic'):	
		u=0.02
		count=0
		if mode=='static':
			dh,dJ=self.CriticalGradient(T)
		elif mode=='dynamic':
			dh,dJ=self.DynamicalCriticalGradient(T)
			
		fit=self.HC
		print(count,fit)
		for i in range(Iterations):
			count+=1
			self.h+=u*dh
			self.J+=u*dJ
			
			Vmax=5
			for i in range(self.size):
				if np.abs(self.h[i])>Vmax:
					self.h[i]=Vmax*np.sign(self.h[i])
				for j in np.arange(i+1,self.size):
					if np.abs(self.J[i,j])>Vmax:
						self.J[i,j]=Vmax*np.sign(self.J[i,j])
						
			if mode=='static':
				dh,dJ=self.CriticalGradient(T)
			elif mode=='dynamic':
				dh,dJ=self.DynamicalCriticalGradient(T)
			fit=self.HC
			
			if count%10==0:
				print(count,fit,np.max(np.abs(self.J)))
			

#Transform bool array into positive integer
def bool2int(x):				
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
        y += j*2**i
    return int(y)
    
#Transform positive integer into bit array
def bitfield(n,size):	
    x = [int(x) for x in bin(int(n))[2:]]
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

	
