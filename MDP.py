import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = np.copy(initialV)
        iterId = 0
        epsilon = np.inf

        while iterId < nIterations and epsilon > tolerance :
            iterId += 1
            Q = self.R + self.discount * np.dot(self.T, V)
            V_old = np.copy(V)
            V = np.max(Q, axis=0)
            epsilon = np.linalg.norm(V - V_old, np.inf)
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        Q = self.R + self.discount * np.dot(self.T, V)
        policy = np.argmax(Q, axis=0)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi
        => R^pi = (I - gamma T^pi) V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        
        R_pi = self.R[policy, np.arange(self.nStates)]
        T_pi = self.T[policy, np.arange(self.nStates), :]
        
        
        A = np.eye(self.nStates) - self.discount * T_pi
        V = np.linalg.solve(A, R_pi)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = np.copy(initialPolicy)
        iterId = 0

        while iterId < nIterations :
            V = self.evaluatePolicy(policy)
            new_policy = self.extractPolicy(V)

            if(policy == new_policy).all():
                break

            policy = new_policy
            iterId += 1

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = np.copy(initialV)
        iterId = 0
        epsilon = np.inf

        # 根據給定的策略，建構 R^pi 和 T^pi
        R_pi = self.R[policy, np.arange(self.nStates)]
        T_pi = self.T[policy, np.arange(self.nStates), :]

        while iterId < nIterations and epsilon > tolerance:
            V_prev = np.copy(V)
            
            # 應用 Bellman 期望方程進行更新
            V = R_pi + self.discount * np.dot(T_pi, V_prev)
            
            epsilon = np.linalg.norm(V - V_prev, np.inf)
            iterId += 1

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = np.copy(initialPolicy)
        V = np.copy(initialV)
        iterId = 0
        epsilon = np.inf

        while iterId < nIterations and epsilon > tolerance:
            V_prev = np.copy(V)

            # 1. 部分策略評估 (只迭代幾次，而不是完全收斂)
            V, _, _ = self.evaluatePolicyPartially(policy, V, nIterations=nEvalIterations)
            
            # 2. 策略改進
            policy = self.extractPolicy(V)
            
            epsilon = np.linalg.norm(V - V_prev, np.inf)
            iterId += 1

        return [policy,V,iterId,epsilon]
        