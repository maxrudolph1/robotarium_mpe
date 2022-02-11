import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
sb.set_theme(style="darkgrid")

bcfc_rew = []
unif_rew = []
mono_rew = []
rand_rew = []

tasks = ['navigation','coverage','transport']
meths = ['expert', 'assigned', 'loc_based', 'uniform', 'combined']
data_dict = {}
diff_dict = {}
for meth in meths:
    data_dict[meth] = {}
    for i,task in enumerate(tasks):
        if meth == 'combined':
            data_dict[meth][task] = np.sum(np.load('./'+meth+'/reward_' + 'combined' + '.npy')[i].squeeze(), axis=0)
        else:
            data_dict[meth][task] =  np.sum(np.load('./'+meth+'/reward_' + task + '.npy').squeeze(), axis=0)
other_meths = [ 'loc_based', 'uniform', 'combined']
for meth in other_meths:
    diff_dict[meth] = {}
    for i, task in enumerate(tasks):
        diff_dict[meth][task] = data_dict['assigned'][task] - data_dict[meth][task]


runs = data_dict['assigned']['navigation'].shape[0]
task_list = []
meth_list = []
val_list = []
diff_task_list = []
diff_meth_list = []
diff_val_list = []
for meth in meths:
    for task in tasks:
        for i in range(runs):
            task_list.append(task)
            meth_list.append(meth)
            val_list.append(data_dict[meth][task][i])

for meth in other_meths:
    for task in tasks:
        for i in range(runs):
            diff_task_list.append(task)
            diff_meth_list.append(meth)
            diff_val_list.append(diff_dict[meth][task][i])


diffs = np.array(diff_val_list)

print(np.sum(diffs >= 0) / len(diff_val_list))

df = pd.DataFrame({'task' : task_list, 'meth': meth_list, 'rew':val_list})
diff_df = pd.DataFrame({'task' : diff_task_list, 'meth': diff_meth_list, 'rew':diff_val_list})


for task in tasks:
    for meth in other_meths:
        U1, p = mannwhitneyu(data_dict[meth][task],data_dict['assigned'][task])
        nx, ny = data_dict[meth][task].shape[0], data_dict['assigned'][task].shape[0]


count = 0
fig, axs = plt.subplots(1,3)
for i,task in enumerate(tasks):
    for j,meth in enumerate(['combined']):
        
        count += 1
        mask = data_dict['assigned'][task] > data_dict[meth][task]
        axs[i].plot(data_dict[meth][task][mask],data_dict['assigned'][task][mask], 'g.')
        axs[i].plot(data_dict[meth][task][np.logical_not(mask)],data_dict['assigned'][task][np.logical_not(mask)], 'r.')
        perf = str(np.sum(mask)/mask.shape[0])
        print(perf)
        yl = axs[i].get_ylim()
        xl = axs[i].get_xlim()
        lim = np.array(xl if xl[1] - xl[0] > yl[1] - yl[0] else yl)
        axs[i].set_xlim(lim[0], lim[1])
        axs[i].set_ylim(lim[0], lim[1])
        axs[i].set_title(perf)
        axs[i].plot(lim, lim, '-')
        axs[i].set_aspect('equal')


## Plotting Violins
'''
plt.figure()
for op,task in enumerate(tasks):
    plt.subplot(1,3,op+1)
    cur_data = df.query("task == '" + task + "'")
    violin_width = 1 if task == 'navigation' else 1
    ax = sb.violinplot(data=cur_data, 
                    x='meth', 
                    y='rew',
                    linewidth=0,
                    label='_nolegend_',
                    width=violin_width)
    ax = sb.stripplot(data=cur_data,
                    x='meth',
                    y='rew',
                    size=1.5)
    boxwidth= 0.075 if task == 'transport' else 0.075
    sb.boxplot(data=cur_data,
                    x='meth',
                    y='rew',
                    width=boxwidth,
                    fliersize=0)   
                  
    #patches = []
    print(len(meths))
    ax.set_xticklabels([''] * len(meths))
    ax.set_xlabel('')
    #patches = []
    patches = []
    # Collect colors of violin plots and make opaque
    for col in [ax.collections[l] for l in [0,2,4,6,8]]: #[0,2,4,6] for 4 different plots
        patches.append(col.get_facecolor())
        print(col.get_facecolor())
        col.set_alpha(.2)
    for col in [ax.collections[l] for l in [9,10,11,12,13]]:#,12,13]]: # [8,9,10,11] for 4 different plots
        col.set_alpha(.3)

    print('----------')  

    patch0 = mpatches.Patch(color=patches[0], label='Expert')
    patch1 = mpatches.Patch(color=patches[1], label='BCFC (Full Pipeline)')
    patch2 = mpatches.Patch(color=patches[2], label='BCFC (Random Task Allocation)')
    patch3 = mpatches.Patch(color=patches[3], label='BCFC (Uniform Task Allocation)')
    patch4 = mpatches.Patch(color=patches[4], label='Monolithic')   

    if task == 'transport':
        plt.legend(handles=[patch0, patch1, patch2, patch3, patch4])
        plt.ylabel('Cumulative Episodic Reward')
    else:
        plt.ylabel('')
        pass#ax.get_legend().remove()


    lab = [task in e.lower() for e in tasks]
    res = [i for i, val in enumerate(lab) if val]
    plt.title(tasks[res[0]])




plt.figure()
for op,task in enumerate(tasks):
    plt.subplot(1,3,op+1)
    cur_data = diff_df.query("task == '" + task + "'")
    violin_width = 1 if task == 'navigation' else 1
    ax = sb.violinplot(data=cur_data, 
                    x='meth', 
                    y='rew',
                    linewidth=0,
                    label='_nolegend_',
                    width=violin_width)
    ax = sb.stripplot(data=cur_data,
                    x='meth',
                    y='rew',
                    size=1.5)
    boxwidth= 0.075 if task == 'transport' else 0.075
    sb.boxplot(data=cur_data,
                    x='meth',
                    y='rew',
                    width=boxwidth,
                    fliersize=0)   
                  
    #patches = []
    print(len(meths))
    ax.set_xticklabels([''] * len(other_meths))
    ax.set_xlabel('')
    #patches = []
    patches = []
    # Collect colors of violin plots and make opaque
    for col in [ax.collections[l] for l in [0,2,4]]: #[0,2,4,6] for 4 different plots
        patches.append(col.get_facecolor())
        print(col.get_facecolor())
        col.set_alpha(.2)
    for col in [ax.collections[l] for l in [6,7,8]]:#,12,13]]: # [8,9,10,11] for 4 different plots
        col.set_alpha(.3)

    print('----------')  


    patch0 = mpatches.Patch(color=patches[0], label='BCFC (Random Task Allocation)')
    patch1 = mpatches.Patch(color=patches[1], label='BCFC (Uniform Task Allocation)')
    patch2 = mpatches.Patch(color=patches[2], label='Monolithic')   

    if task == 'transport':
        plt.legend(handles=[patch0, patch1, patch2])
        plt.ylabel('Cumulative Episodic Reward')
    else:
        plt.ylabel('')
        pass#ax.get_legend().remove()


    lab = [task in e.lower() for e in tasks]
    res = [i for i, val in enumerate(lab) if val]
    plt.title(tasks[res[0]])
'''
plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') 

