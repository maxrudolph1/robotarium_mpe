import joblib
import numpy as np
class RewardNet:
    def __init__(self, gamma=0.99, reward_path=None, policy_path=None, model_choice=0):
        loaded_params = joblib.load(reward_path if policy_path is None else policy_path)
        
        hidden_size=128
        if reward_path is not None:
            
            reward_params = loaded_params
            print('Input Size: ' + str(reward_params[0].shape[0]))
            self.reww1 = np.transpose(reward_params[0])
            self.rewb1 = np.reshape(reward_params[1], (hidden_size, 1))
            
            self.reww2 = np.transpose(reward_params[2])
            self.rewb2 = np.reshape(reward_params[3], (hidden_size, 1))

            self.rewwo = np.transpose(reward_params[4])
            self.rewbo = reward_params[5]

            self.vfnw1 = np.transpose(reward_params[6])
            self.vfnb1 = np.reshape(reward_params[7], (hidden_size, 1))
            
            self.vfnw2 = np.transpose(reward_params[8])
            self.vfnb2 = np.reshape(reward_params[9], (hidden_size, 1))

            self.vfnwo = np.transpose(reward_params[10])
            self.vfnbo = reward_params[11]

            self.gamma = gamma
        if policy_path is not None:
            policy_params = loaded_params
            self.params_dict = {}
            self.params_dict['w0'] = np.transpose(policy_params[0 + 12*model_choice])
            self.params_dict['b0'] = np.reshape(policy_params[1 + 12*model_choice], (hidden_size, 1))
            self.params_dict['w1'] = np.transpose(policy_params[2 + 12*model_choice])
            self.params_dict['b1'] = np.reshape(policy_params[3 + 12*model_choice], (hidden_size, 1))
            self.params_dict['wo'] = np.transpose(policy_params[4 + 12*model_choice])
            self.params_dict['bo'] = np.reshape(policy_params[5 + 12*model_choice], (5, 1))
            print([x.shape for x in policy_params])

    def relu(self, x):
        x[x < 0] = 0
        
        #x = np.tanh(x)
        return x

    def vfn(self, x):
        y1 = self.relu(np.matmul(self.vfnw1, x) + self.vfnb1)
        y2 = self.relu(np.matmul(self.vfnw2, y1) + self.vfnb2)
        yo = self.relu(np.matmul(self.vfnwo, y2) + self.vfnbo)
        return yo
    
    def rew(self, x):
        y1 = self.relu(np.matmul(self.reww1, x) + self.rewb1)
        y2 = self.relu(np.matmul(self.reww2, y1) + self.rewb2)
        yo = self.relu(np.matmul(self.rewwo, y2) + self.rewbo)
        #print(y1, y2, yo)
        return yo

    def calc_reward(self, obs, obs_n):
        if len(obs.shape) == 1:
            obs = obs[:, np.newaxis]
            obs_n = obs_n[:, np.newaxis]
        re = self.rew(obs)
        vf = self.vfn(obs)
        vfn = self.vfn(obs_n)
        yo = re + vfn * self.gamma - vf #self.rew(obs) + self.vfn(obs_n) * self.gamma - self.vfn(obs)
        return yo, re, vf #self.vfn(obs)#yo
    
    def get_action(self, obs):
        # Returns raw output of network
        obs = np.array(obs)
        obs = obs[:, np.newaxis]

        y1 = self.relu(np.matmul(self.params_dict['w0'], obs) +self.params_dict['b0'])
        y2 = self.relu(np.matmul(self.params_dict['w1'], y1) + self.params_dict['b1'])
        yo = (np.matmul(self.params_dict['wo'], y2) + self.params_dict['bo'])
        return np.array(yo).squeeze()

# def main():
#     path ='/home/mrudolph/Documents/multi_agent_learning/expert/learned_rewards/simple_navigation/d_1_00100' #'/home/mrudolph/Documents/multi_agent_learning/inverse_rl/log/airl/simple_transport/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/d_0_02000'
#     reward = RewardNet(path)
#     obs = np.random.random(size=(14,)).astype(np.float64)
#     rew = reward.calc_reward(obs).squeeze()
#     print(rew)
    
# if __name__=="__main__":
#     main()