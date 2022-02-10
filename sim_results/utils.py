import json
import numpy as np
def find_traits(task_name=None, team_idx=0):
    if task_name =='nav':
        task_idx=2
    elif task_name=='cov':
        task_idx=0
    else:
        task_idx=1
    f = open('/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json')
    data = json.load(f)
    Q = np.array(data[str(team_idx)]['Q'])
    X = np.array(data[str(team_idx)]['X'])
    ass_agents = []
    ass_agents = X[task_idx, :]
    
    traits = []
    for i,n in enumerate(ass_agents):
        for _ in range(int(n)):
            traits.append(Q[i, task_idx])
    return np.array(traits).squeeze()

def find_all_agent_traits(trait=None, team_idx=0):
    f = open('/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json')
    data = json.load(f)
    Q = np.array(data[str(team_idx)]['Q'])
    X = np.array(data[str(team_idx)]['X'])

    if trait =='speed':
        trait_idx=2
    elif trait=='sensing':
        trait_idx=0
    else:
        trait_idx=1

    trait_list = []

    for i in range(3):
        for j in range(3):
            trait_list.append(Q[i, trait_idx])
    return np.array(trait_list).squeeze()