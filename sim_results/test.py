from utils import *

l = 1
speed = find_traits(task_name='nav', team_idx=l%20) #if task == 'navigation' else np.ones((N,))
sensing_rad = find_traits(task_name='cov', team_idx=l%20) #if task == 'coverage' else np.ones((N,))
payload_cap = find_traits(task_name='trans', team_idx=l%20) # if task == 'navigation' else np.ones((N,))

print([find_all_agent_traits(trait='speed', team_idx = i ) for i in range(20)])

for i in range(20):
    print(find_traits(task_name='cov', team_idx=i%20))    
