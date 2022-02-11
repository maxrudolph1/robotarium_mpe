import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import json
import numpy as np
import matplotlib.patches as mpatches
sns.set_theme(style="darkgrid")

f = open('data_dict.json')
data = json.load(f)

meth_list = list(data.keys())

env_list = list(data[meth_list[1]].keys())
idx = (np.ones((len(data[meth_list[1]][env_list[0]]), 50)) * np.arange(50)[np.newaxis,:]).flatten().tolist()
traj_idx = []

for i in range(len(data[meth_list[1]][env_list[0]])):
    traj_idx.extend([str(i)] * 50)

pd_data_list = []
pd_method_list = []
pd_env_list = []
idx_list = []
traj_list = []

for j,env in enumerate(env_list):
    for i,meth in enumerate(meth_list):
        data_arr = np.array(data[meth][env]).flatten().tolist()  
        pd_data_list.extend(data_arr)
        pd_method_list.extend([meth] * len(data_arr) )
        pd_env_list.extend([env] * len(data_arr) )

        
        idx_list.extend(idx)
        traj_list.extend(traj_idx)

df = pd.DataFrame(data={'reward' : pd_data_list,
                        'method' : pd_method_list,
                        'env' : pd_env_list,
                        'idx': idx_list,
                        'traj' : traj_list})
env_name_list = ['Navigation', 'Coverage', 'Transport']

# for env in env_list:
#     plt.figure()
#     plt.title(env)
#     ax = sns.lineplot(data=df.query("method != 'expert' and env == '" + env + "'"),
#                 y='reward',
#                 x='idx',
#                 hue='method',
#                 label='_nolegend_')
#     ax.get_legend().remove()
#     patches = []
#     for col in ax.collections[0:6]:
#         color = col.get_facecolor()
#         color[0][3] = 1
#         patches.append(color)

#     patch1 = mpatches.Patch(color=patches[0].squeeze(), label='BCFC (ours)')
#     patch2 = mpatches.Patch(color=patches[1], label='Location-based Assignment')
#     patch3 = mpatches.Patch(color=patches[2], label='Uniform Assignment')
#     patch4 = mpatches.Patch(color=patches[3], label='Monolithic')
#     if env == 'trans':
#         plt.legend(handles=[patch1, patch2, patch3, patch4])
#         plt.ylabel('Reward')
#     else:
#         plt.ylabel('')
#         pass#ax.get_legend().remove()
#     plt.xlabel('Time Steps')

#     lab = [env in e.lower() for e in env_name_list]
#     res = [i for i, val in enumerate(lab) if val]
#     plt.title(env_name_list[res[0]])
#     plt.savefig('figs/svg/' + env + '_line.svg')
#     plt.savefig('figs/png/' + env + '_line.png')



cum_data = []
env_tag_list = []
meth_tag_list = []

for env in env_list:
    for meth in meth_list:
        for i in range(200):
            #rews = df.query("method != 'expert' and env == '" + env + "' and method == '" + meth + "' and traj == '" + str(i) + "'")['reward'].tolist()
            rews = df.query("env == '" + env + "' and method == '" + meth + "' and traj == '" + str(i) + "'")['reward'].tolist()
            cum_data.append(np.sum(rews))    
            env_tag_list.append(env)
            meth_tag_list.append(meth)

        print(meth)

violin_df = pd.DataFrame(data={'reward' : cum_data,
                                'env' : env_tag_list,
                                'method' : meth_tag_list})



env_name_list = ['Navigation', 'Coverage', 'Transport']
plt.figure()
patches = []
for ol,env in enumerate(env_list):
    plt.subplot(1,3,ol+1)
    #plt.title(env)
    #cur_data = violin_df.query("method != 'expert' and env == '" + env + "'")
    cur_data = violin_df.query("env == '" + env + "'")
    violin_width = 1 if env == 'nav' else 1
    ax = sns.violinplot(data=cur_data, 
                    x='method', 
                    y='reward',
                    linewidth=0,
                    label='_nolegend_',
                    width=violin_width)
    ax = sns.stripplot(data=cur_data,
                    x='method',
                    y='reward',
                    size=1.5)
    boxwidth= 0.025 if env == 'trans' else 0.075
    sns.boxplot(data=cur_data,
                    x='method',
                    y='reward',
                    width=boxwidth,
                    fliersize=0)                 
                    
    #patches = []
    ax.set_xticklabels([''] * 5)
    ax.set_xlabel('')
    
    # for col in ax.collections[0:5]:
    #     patches.append(col.get_facecolor())
    #     col.set_alpha(0.5)
        # Collect colors of violin plots and make opaque

    for col in [ax.collections[l] for l in [0,2,4,6,8]]: #[0,2,4,6] for 4 different plots
        if len(patches) < 5:
            patches.append(col.get_facecolor())
        col.set_alpha(.2)
    # for col in [ax.collections[l] for l in [9,10,11,12,13]]:#,12,13]]: # [8,9,10,11] for 4 different plots
    #     col.set_alpha(.3)
    
    patch0 = mpatches.Patch(color=patches[0], label='Expert')
    patch1 = mpatches.Patch(color=patches[1], label='BCFC (Full Pipeline)')
    patch2 = mpatches.Patch(color=patches[2], label='BCFC (Random Task Allocation)')
    patch3 = mpatches.Patch(color=patches[3], label='BCFC (Uniform Task Allocation)')
    patch4 = mpatches.Patch(color=patches[4], label='Monolithic')   

    if env == 'trans':
        plt.legend(handles=[patch0, patch1, patch2, patch3, patch4])
        plt.ylabel('Cumulative Episodic Reward')
    else:
        plt.ylabel('')
        pass#ax.get_legend().remove()


    lab = [env in e.lower() for e in env_name_list]
    res = [i for i, val in enumerate(lab) if val]
    plt.title(env_name_list[res[0]])

    plt.ylabel('Cumulative Episodic Reward')
    plt.savefig('figs/svg/' + env + '_violin.svg')
    plt.savefig('figs/png/' + env + '_violin.png')
    

plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') 
