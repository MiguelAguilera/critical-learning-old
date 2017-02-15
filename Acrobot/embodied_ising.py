import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import gym


class ising:
	#Initialize the network
	def __init__(self, netsize,Nsensors=1,Nmotors=1):	#Create ising model
	
		self.size=netsize
		self.Ssize=Nsensors			#Number of sensors
		self.Msize=Nmotors			#Number of sensors
		
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.max_weights=1
		self.randomize_state()
		
		self.env = gym.make('Acrobot-v1')
		mass=1.8
		self.set_mass(mass)
		self.observation = self.env.reset()
		self.randomize_position()
		
		self.Beta=1.0
		self.defaultT=max(100,netsize*20)
		self.maxspeed=self.env.MAX_VEL_1
		self.minspeed=-self.env.MAX_VEL_1
		self.maxspeed2=self.env.MAX_VEL_2
		self.minspeed2=-self.env.MAX_VEL_2
		self.maxacc=5
		self.minacc=-5
		self.maxheight=2
		self.minheight=-2
#		self.Ssize1=int(np.floor(self.Ssize/2))
		self.sensorbins=np.linspace(-1.01,1.01,2**(self.Ssize)+1)
			
		self.Update(0)
		
		
	def get_state(self,mode='all'):
		if mode=='all':
			return self.s
		elif mode=='motors':
			return self.s[-self.Msize:]
		elif mode=='sensors':
			return self.s[0:self.Ssize]
	
	def get_state_index(self,mode='all'):
		return bool2int(0.5*(self.get_state(mode)+1))
			
	def set_mass(self,mass=1):
		self.env.LINK_MASS_1=mass
		self.env.LINK_MASS_2=mass
	#Randomize the state of the network
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1

	def randomize_position(self):
		self.observation = self.env.reset()
		self.theta1_dot=0
		self.env.state = self.env.np_random.uniform(low=-np.pi, high=np.pi, size=(4,))
		
	#Set random bias to sets of units of the system
	def random_fields(self,max_weights=None):
		if max_weights is None:
			max_weights=self.max_weights
		self.h[self.Ssize:]=max_weights*(np.random.rand(self.size-self.Ssize)*2-1)		
		
	#Set random connections to sets of units of the system
	def random_wiring(self,max_weights=None):	#Set random values for h and J
		if max_weights is None:
			max_weights=self.max_weights
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				if i<j and (i>=self.Ssize or j>=self.Ssize):
					self.J[i,j]=(np.random.randn(1)*2-1)*self.max_weights
		
	def Move(self):
		action=int(np.digitize(np.sum(self.s[-self.Msize:])/self.Msize,[-1/3,1/3,1.1]))
		observation, reward, done, info = self.env.step(action)
		theta1_dot_p=self.theta1_dot
		self.theta1_dot=self.env.state[2]
		self.theta1_dot2=self.theta1_dot-theta1_dot_p
		
	def SensorIndex(self,x,xmax):
		return np.digitize(np.clip(x,-xmax,xmax)/xmax,self.sensorbins)-1
		
	def UpdateSensors(self):
		self.acc=self.theta1_dot2
		s = self.env.state
		self.theta=s[0]
		self.ypos=-np.cos(s[0]) - np.cos(s[1] + s[0])
		self.xpos=np.sin(s[0]) + np.sin(s[1] + s[0])
		
		self.speed=self.env.state[2]
		self.speedx=self.speed*np.cos(s[0])
		self.speedy=self.speed*np.sin(s[0])
		
		self.speed2=self.env.state[3]
		
		self.posx_ind=self.SensorIndex(self.xpos,self.maxheight)
		self.posy_ind=self.SensorIndex(self.ypos,self.maxheight)
		
		self.speed_ind=self.SensorIndex(self.speed,self.maxspeed)
		self.speedx_ind=self.SensorIndex(self.speedx,self.maxspeed)
		self.speedy_ind=self.SensorIndex(self.speedy,self.maxspeed)
		self.speed2_ind=self.SensorIndex(self.speed2,self.maxspeed2)
		
		self.acc_ind=self.SensorIndex(self.acc,self.maxacc)
		self.accx_ind=self.SensorIndex(self.acc*np.cos(s[0]),self.maxacc)
		self.accy_ind=self.SensorIndex(self.acc*np.sin(s[0]),self.maxacc)
		
#		self.s[0:self.Ssize]=self.code[self.sensor_ind,:]
#		self.s[self.Ssize1:self.Ssize]= 2*bitfield(self.speed_ind,self.Ssize-self.Ssize1)-1
#		self.s[0:self.Ssize1]=2*bitfield(self.sensor1_ind,self.Ssize1)-1
#		
#		self.s[self.Ssize1:self.Ssize]= 2*bitfield(self.accx_ind,self.Ssize-self.Ssize1)-1

		self.s[0:self.Ssize]=2*bitfield(self.acc_ind,self.Ssize)-1
		
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
		
		self.y=np.zeros(T)
		
		self.randomize_state()
		self.randomize_position()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.SequentialUpdate()
			self.y[t]=self.ypos
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
		
		dh[0:self.Ssize]=0
		dJ[0:self.Ssize,0:self.Ssize]=0
		
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
		
		self.y=np.zeros(T)
		
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.SequentialUpdate()
			self.y[t]=self.ypos
#			self.env.render()
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
					dJ[j,i]+=dJ1[j,i]*0.5
				elif j>i:
					dJ[i,j]+=dJ1[j,i]*0.5
		
		self.HCl=mG-msH*mF
		self.HC=np.sum(self.HCl[self.Ssize:])
		
		
		return dh,dJ
		
		
	def CriticalLearning(self,Iterations,T=None,mode='dynamic',reset=False,verbosity=True):	
		u=0.02
		count=0
		if mode=='static':
			dh,dJ=self.CriticalGradient(T)
		elif mode=='dynamic':
			dh,dJ=self.DynamicalCriticalGradient(T)
			
		self.set_mass(2)
		fit=self.HC
		if verbosity:
			print(count,fit,np.max(np.abs(self.J)),np.mean(np.abs(self.J[0:self.Ssize,self.Ssize:])),np.mean(self.y),np.max(self.y))
		
		self.l2=0.1
		for i in range(Iterations):
#			if i%(Iterations/20)==0:
#				mass=np.clip(0.5+2*i/Iterations,1,2)
#				self.set_mass(mass)
#			if i%10==0:
			if reset:
				self.randomize_state()
				self.randomize_position()

			count+=1
			self.h+=u*(dh - self.l2*self.h)
			self.J+=u*(dJ - self.l2*self.J)
			
			Vmax=1
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
			
#			if np.max(self.y)>1.5:
#				break
			if count%1==0 and verbosity:
				print(count,fit,self.env.LINK_MASS_1,np.max(np.abs(self.J)),np.mean(np.abs(self.J[0:self.Ssize,self.Ssize:])),np.mean(self.y),np.max(self.y))
#		self.set_mass(2)
			

#Transform bool array into positive integer
def bool2int(x):				
	y = 0
	for i,j in enumerate(np.array(x)[::-1]):
#		y += j<<i
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

	
