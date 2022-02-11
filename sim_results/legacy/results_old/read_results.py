import numpy as np
from matplotlib import pyplot as plt

bcfc_rew = []
unif_rew = []
mono_rew = []
rand_rew = []

tasks = ['navigation','coverage', 'transport']
meths = ['assigned', 'loc_based', 'uniform', 'combined']
data_dict = {}

for meth in meths:
    data_dict[meth] = {}
    for i,task in enumerate(tasks):
        if meth == 'combined':
            data_dict[meth][task] = np.load('./'+meth+'/reward_' + 'combined' + '.npy')[i].squeeze()
        else:
            data_dict[meth][task] =  np.load('./'+meth+'/reward_' + task + '.npy').squeeze()


# c = ['r-', 'k-', 'b-', 'g-']
    
# for i, meth in enumerate(meths):
#     plt.plot((data_dict[meth]['coverage']), c[i])
#     plt.legend(meths)



for i,task in enumerate(tasks):
    plt.subplot(1,3,i+1)
    
    plt.boxplot([np.sum(data_dict[meth][task], axis=0) for meth in meths])
    plt.xticks(np.arange(4) + 1, meths)
    plt.title(task)
    plt.xlabel('method')
    plt.ylabel('reward')
plt.show()