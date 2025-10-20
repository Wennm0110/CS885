import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        rewards_per_episode -- A list containing the cumulative discounted reward for each episode
        '''

        Q = np.copy(initialQ)
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        
        # 新增：用於儲存每個 episode 的獎勵
        rewards_per_episode = []

        for episode in range(nEpisodes):
            state = s0
            # 新增：用於計算當前 episode 的累積折扣獎勵
            cumulative_discounted_reward = 0.0

            for step in range(nSteps):
                # 1. 選擇動作 (ε-greedy)
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    if temperature > 0:
                        q_values = Q[:, state]
                        q_values_stable = q_values - np.max(q_values)
                        probabilities = np.exp(q_values_stable / temperature)
                        probabilities /= np.sum(probabilities)
                        action = np.random.choice(self.mdp.nActions, p=probabilities)
                    else:
                        action = np.argmax(Q[:, state])

                # 2. 執行動作並觀察結果
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                cumulative_discounted_reward += (self.mdp.discount ** step) * reward

                # 3. 更新 Q-value
                N[action, state] += 1
                alpha = 1.0 / N[action, state]

                max_q_next_state = np.max(Q[:, nextState])
                
                td_target = reward + self.mdp.discount * max_q_next_state
                Q[action, state] += alpha * (td_target - Q[action, state])

                # 4. 更新狀態
                state = nextState
            
            # 將這個 episode 的總獎勵記錄下來
            rewards_per_episode.append(cumulative_discounted_reward)

        # 導出最終策略
        policy = np.argmax(Q, axis=0)

        # 回傳 Q 表、策略以及每個 episode 的獎勵歷史
        return [Q, policy, rewards_per_episode]