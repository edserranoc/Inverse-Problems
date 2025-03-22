import numpy as np
from typing import Tuple
import pytwalk

class Forward_mapping:
    def __init__(self,sample_size):
        self.sample_size = sample_size
        
    def logpdf(self, x):
        return -10*(x[0]**2-x[1])**2 - (x[1]-1/4)**2
    
    def logalpha(self, x, y):
        return min(0, self.logpdf(y) - self.logpdf(x))
    
    def energy(self, x):
        return -self.logpdf(x)
    
    def support(self, x):
        rt = True
        rt&= -2 < x[0] < 2
        rt&= -1 < x[1] < 2
        return rt
    
    def init(self):
        x = np.zeros(2)
        x[0] = np.random.uniform(-2, 2)
        x[1] = np.random.uniform(-1, 2)
        return x
    
    def mcmc_normal_proposal(self, 
                             x0:np.ndarray,
                             gamma:float)->Tuple[np.ndarray, np.ndarray, int]:
        
        """Compute the MCMC chain using a normal proposal distribution
        
        :param x0: The initial state of the chain (a 2D vector)
        :param K: The number of iterations of the chain (a scalar)
        :return: The MCMC chain (a Kx2 matrix) and the acceptance rate (a scalar)
        """
        
        x = x0
        chain = np.zeros((self.sample_size, 2))
        energy = np.zeros(self.sample_size)
        accepted = 0
        
        chain[0] = x   
        energy[0] = self.energy(x)
        
        for k in range(1, self.sample_size):
            
            y = np.random.multivariate_normal(x, gamma**2*np.eye(2))
            if np.log(np.random.uniform()) < self.logalpha(x, y):
                x = y
                accepted += 1
            energy[k] = -self.logpdf(x)
            chain[k] = x
        
        return chain, energy, accepted/self.sample_size
    
    def run_twalk(self):
        dist_mcmc = pytwalk.pytwalk(n= 2, 
                                U= self.energy, 
                                Supp =self.support)
        x0 = self.init()
        xp0 = self.init()
        print("Initial states t-walk: ", x0,", ", xp0)
        
        dist_mcmc.Run(T=    self.sample_size, 
                  x0=   x0,
                  xp0=  xp0)
        return dist_mcmc