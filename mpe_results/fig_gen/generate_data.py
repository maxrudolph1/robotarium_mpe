import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json
import seaborn as sb

nav_expert_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/expert/simple_navigation/simple_navigation_reward.pkl'
cov_expert_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/expert/simple_coverage/simple_coverage_reward.pkl'
trans_expert_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/expert/simple_transport/simple_transport_reward.pkl'

nav_target_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/target_team/simple_navigation/simple_navigation_reward.pkl'
cov_target_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/target_team/simple_coverage/simple_coverage_reward.pkl'
trans_target_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/target_team/simple_transport/simple_transport_reward.pkl'

nav_random_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/random_team/simple_navigation/simple_navigation_reward.pkl'
cov_random_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/random_team/simple_coverage/simple_coverage_reward.pkl'
trans_random_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/random_team/simple_transport/simple_transport_reward.pkl'

nav_uniform_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/uniform_team/simple_navigation/simple_navigation_reward.pkl'
cov_uniform_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/uniform_team/simple_coverage/simple_coverage_reward.pkl'
trans_uniform_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/uniform_team/simple_transport/simple_transport_reward.pkl'

combined_path = '/home/mrudolph/Documents/multi_agent_learning/expert/reward_performance/no_ass_team/simple_combined/simple_combined_reward.pkl'
expert_paths = [nav_expert_path,cov_expert_path,trans_expert_path]
target_team_paths = [nav_target_path,cov_target_path,trans_target_path]
random_team_paths = [nav_random_path, cov_random_path, trans_random_path]
uniform_team_paths = [nav_uniform_path, cov_uniform_path, trans_uniform_path]

plt.figure()


env_ord = ['nav', 'cov', 'trans']

data = {}
data['expert'] = {}

for i,path in enumerate(expert_paths):
    with open(path, "rb") as f:
        reward_data = pkl.load(f)

        reward_data = reward_data[:,:,0]
    data['expert'][env_ord[i]] = reward_data.tolist()
    plt.subplot(1,3,i+1)
    cumulative_rew = np.cumsum(reward_data, axis=1)
    cumulative_rew = reward_data

    avg_rew = np.mean(cumulative_rew, axis=0)
    #plt.plot(cumulative_rew.T, 'r-')
    #plt.plot(avg_rew, 'k-')
    plt.title('Expert Team')
    
data['BCFC (ours)'] = {}
for i,path in enumerate(target_team_paths):
    
    with open(path, "rb") as f:
        reward_data = pkl.load(f)
        reward_data = reward_data[:,:,0]
    data['BCFC (ours)'][env_ord[i]] = reward_data.tolist()
    plt.subplot(1,3,i+1)
    cumulative_rew = np.cumsum(reward_data, axis=1)
    cumulative_rew = reward_data
    avg_rew = np.mean(cumulative_rew, axis=0)

    plt.plot(avg_rew, 'k-')
    plt.title('Target Team')

data['Assignment-agnostic']  = {}
for i,path in enumerate(random_team_paths):
    with open(path, "rb") as f:
        reward_data = pkl.load(f)
        reward_data = reward_data[:,:,0]
    data['Assignment-agnostic'][env_ord[i]] = reward_data.tolist()
    plt.subplot(1,3,i+1)
    cumulative_rew = np.cumsum(reward_data, axis=1)
    cumulative_rew = reward_data
    avg_rew = np.mean(cumulative_rew, axis=0)
    # plt.plot(cumulative_rew.T, 'b-')
    plt.plot(avg_rew, 'k-')
    # plt.plot(cumulative_rew.T, 'k-', linewidth=0.3)
    plt.title('Target Team')

aa_vals = cumulative_rew
data['Uniform Assignment']  = {}
for i,path in enumerate(uniform_team_paths):
    with open(path, "rb") as f:
        reward_data = pkl.load(f)
        reward_data = reward_data[:,:,0]
    data['Uniform Assignment'][env_ord[i]] = reward_data.tolist()
    plt.subplot(1,3,i+1)
    cumulative_rew = np.cumsum(reward_data, axis=1)
    cumulative_rew = reward_data - data['BCFC (ours)'][env_ord[i]]
    print(cumulative_rew.shape)
    avg_rew = np.mean(cumulative_rew, axis=0)
    # plt.plot(cumulative_rew.T, 'b-')
    #plt.plot(cumulative_rew.T, 'b-', linewidth=0.1)
    plt.plot(avg_rew, 'b-')
    plt.title('Target Team')
    

ua_vals = cumulative_rew
data['Unstructured Learning'] = {}
with open(combined_path, "rb") as f:
    reward_data = pkl.load(f)
traj_num = reward_data.shape[0]
traj_len = reward_data.shape[1]
trans_rew = np.zeros((traj_num, traj_len))
nav_rew = np.zeros((traj_num, traj_len))
cov_rew = np.zeros((traj_num, traj_len))
combined_rew_list = [nav_rew, cov_rew, trans_rew]

for i in range(traj_num):
    for j in range(traj_len):
        sorted_rews = np.sort(reward_data[i,j,:])
        trans_rew[i, j] = sorted_rews[-1]
        cov_rew[i, j] = sorted_rews[0]
        nav_rew[i,j] = sorted_rews[4]

for i,reward in enumerate(combined_rew_list):
    plt.subplot(1,3,i+1)
    cumulative_rew = np.cumsum(reward, axis=1)
    cumulative_rew = reward  -  data['BCFC (ours)'][env_ord[i]]
    avg_rew = np.mean(cumulative_rew, axis=0)
    #plt.plot(cumulative_rew.T, 'b-')
    data['Unstructured Learning'][env_ord[i]] = reward.tolist()
    plt.plot(avg_rew, 'r-')
    #plt.plot(cumulative_rew.T, 'r-', linewidth=0.1)
    plt.title('Target Team')

ul_vals = cumulative_rew
with open('data_dict.json', 'w') as outfile:
    json.dump(data, outfile, indent=3, sort_keys=False)




plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') 