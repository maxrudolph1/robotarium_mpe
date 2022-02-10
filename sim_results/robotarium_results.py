#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from reward_net import RewardNet
from utils import find_traits
import sys
import click

@click.command()
@click.option('--policy_path', type=click.STRING)
@click.option('--save_path', default=None)
@click.option('--task', type=click.Choice(['transport', 'combined', 'navigation', 'coverage']), default='transport')
@click.option('--show', default=False)
@click.option('--runs', default=1)

def main(task, save_path, policy_path, show, runs):
    iterations = 1600
    rewards = np.zeros((3 if task == 'combined' else 1, iterations, runs))
    for l in range(runs):
        # Experiment constants
        
        dt = 1
        N = 9 if task == 'combined' else 3

        #Limit maximum linear speed of any robot
        magnitude_limit = 100
        if task == 'combined':
            init_conditions = np.random.random((3, N))
            init_conditions[0, :] = init_conditions[0, :] * 2 - 3
            init_conditions[1, :] = init_conditions[1, :] * 2 - 1
            init_conditions[2, :] = init_conditions[2, :] * 2 * np.pi

        r = robotarium.Robotarium(number_of_robots=N, show_figure=show, sim_in_real_time=show, initial_conditions = (init_conditions if task=='combined' else np.array([])))
        if task == 'combined':
            si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary( magnitude_limit=0.4, boundary_points = np.array([-4, 4, -1.0, 1.0]))
        else:
            si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary( magnitude_limit=0.4, boundary_points = np.array([-1.6, 1.6, -1.0, 1.0]))

        si_to_uni_dyn = create_si_to_uni_dynamics()
        dxi = np.zeros((2, N))

        speed = find_traits(task_name='nav', team_idx=l%20) #if task == 'navigation' else np.ones((N,))
        sensing_rad = 1/find_traits(task_name='cov', team_idx=l%20) #if task == 'coverage' else np.ones((N,))
        payload_cap = find_traits(task_name='trans', team_idx=l%20) # if task == 'navigation' else np.ones((N,))

        policy = [RewardNet(policy_path=policy_path, model_choice=i, split=False) for i in range(N)]

        obs_size = 9 if task == 'transport' else (39 if task == 'combined' else 14)
        

        if task == 'combined':
            prev_pay = np.ones((9,))
            nav_entity = np.random.random(size=(3,2)) * 2 - 1
            nav_entity[:, 0] -= 2
            cov_entity =  np.random.uniform(-0.75, 0.75, size=(2,2))
            cov_var = np.random.uniform(0.5, 1.5, size=(2,))
            trans_entity = np.array([[0.5, -0.5], [0,0]]).T
            trans_entity[:,0] += 2
    
            
        else:
            prev_pay = np.ones(( 3,))
            nav_entity = np.array([[1,0,-1], [0,0,0]]).T
            nav_entity = np.random.random(size=(3,2)) * 2 - 1
            #cov_entity = np.array([[-1,1], [0,0]])
            cov_entity =  np.random.uniform(-0.75, 0.75, size=(2,2))
            cov_var = np.random.uniform(0.5, 1.5, size=(2,)) #np.random.random(size=(2,)) 
            trans_entity = np.array([[0.5, -0.5], [0,0]]).T

        if show and (task == 'navigation'):
            r.axes.scatter(nav_entity[0,0],nav_entity[0,1], s=2000, zorder=-2)
            r.axes.scatter(nav_entity[1,0],nav_entity[1,1], s=2000, zorder=-2)
            r.axes.scatter(nav_entity[2,0],nav_entity[2,1], s=2000, zorder=-2)
        elif show and (task == 'transport'):
            r.axes.scatter(-0.5,0, s=2000, zorder=-2)
            r.axes.scatter(0.5,0, s=2000, zorder=-2)
        elif show and ((task == 'coverage') or (task == 'combined')):
            r.axes.scatter(cov_entity[0,0],cov_entity[0,1], s=2000, zorder=-2)
            r.axes.scatter(cov_entity[1,0],cov_entity[1,1], s=2000, zorder=-2)
            


        for k in range(iterations):
            
            # Get the poses of the robots
            x = r.get_poses()

            x_ord = np.argsort(x[0, :])



            # Initialize a velocity vector
            F = np.zeros((2,N))
            for i in range(N):
                # Navigation Observation Space
                if task == 'navigation':
                    obs = np.zeros((14,))
                    obs[:2] = np.squeeze(dxi[:, i])
                    obs[2:4] = np.squeeze(x[:2, i])
                    obs[4:10] = (nav_entity - x[:2, [i]].T).flatten()
                    obs[10:] = (x[:2, [l for l in range(N) if not (l == i)]] - x[:2,i]).flatten()
                
                # Transport Observation Space
                if task == 'transport':
                    
                    obs = np.zeros((9,))
                    obs[:2] = np.squeeze(dxi[:, i])
                    obs[2:4] = np.squeeze(x[:2, i])
                    if (prev_pay[i] == 1) and (np.linalg.norm(x[:2, i] - np.array([-0.5,0])) < 0.2):
                        prev_pay[i] = 0
                    elif (prev_pay[i] == 0) and (np.linalg.norm(x[:2, i] - np.array([0.5,0])) < 0.2):
                        prev_pay[i] = 1
                    obs[4] = prev_pay[i]
                    obs[5:] = np.array([0.5, 0, -0.5, 0])

                # Coverage Observation Space
                if task == 'coverage':
                    obs = np.zeros((14,))
                    cov1 = cov_entity[:,0]
                    cov2 = cov_entity[:,1]
                    var1 = cov_var[0]
                    var2 = cov_var[1]
                    obs[:2] = cov1 - x[:2, i]
                    obs[2] = var1
                    obs[3:5] = cov2 - x[:2, i]
                    obs[5] = var2
                    obs[6:10] = (x[:2, [l for l in range(N) if not (l == i)]] - x[:2,i]).flatten()
                    
                    obs[10:12] = np.squeeze(x[:2, i])
                    obs[12:] = np.squeeze(dxi[:, i])


                if task == 'combined':
                    obs = np.zeros((39,))
                    obs[:2] = np.squeeze(x[:2, i])
                    obs[2:4] = np.squeeze(dxi[:, i])
                    if (prev_pay[i] == 1) and (np.linalg.norm(x[:2, i] - np.array([-0.5,0])) < 0.2):
                        prev_pay[i] = 0
                    elif (prev_pay[i] == 0) and (np.linalg.norm(x[:2, i] - np.array([0.5,0])) < 0.2):
                        prev_pay[i] = 1
                    obs[4] = prev_pay[i]
                    obs[5:9] = (trans_entity - x[:2, [i]].T).flatten()
                    obs[9:15] = (nav_entity - x[:2, [i]].T).flatten()
                    obs[15:19] = (cov_entity - x[:2, [i]].T).flatten()
                    obs[19:21] = cov_var
                    obs[21:] =  (x[:2, [l for l in range(N)]] - x[:2,[i]]).flatten()


                act = np.argmax(policy[i].get_action(obs))

                if act == 1:
                    F[0, i] = 1
                elif act == 2:
                    F[0, i] = -1
                elif act == 3:
                    F[1, i] = 1
                elif act == 4:
                    F[1, i] = -1
                
            exp_speed = np.ones((N,))
            exp_speed[x_ord[:3]] = speed

            if task == 'combined':
                dxi += F * dt * exp_speed 
            else:
                dxi += F * dt * speed 
    
            # Make sure that the robots don't collide

            dxi = si_barrier_cert(dxi, x[:2, :])

            # Transform the single-integrator dynamcis to unicycle dynamics
            dxu = si_to_uni_dyn(dxi, x)
            
            # Set the velocities of the robots
            r.set_velocities(np.arange(N), dxu)
            # Iterate the simulation
            r.step()

            if task == 'combined':
                rewards[0, k, l] = navigation_reward(x[:, x_ord[:3]], nav_entity)
                
                rewards[1, k, l] = coverage_reward(x[:, x_ord[3:6]], sensing_rad, cov_entity, cov_var, 3)
                # for u in range(3):
                #     if (k % 100 == 0):
                #         r.axes.scatter(x[0, x_ord[u + 3]],x[1, x_ord[u + 3]], s=20, zorder=2)
                
                rewards[2, k, l] = transport_reward(x[:, x_ord[6:]], prev_pay[x_ord[6:]], payload_cap)
                
            elif task == 'transport':
                rewards[0,k,l] = transport_reward(x, prev_pay, payload_cap)
            elif task == 'coverage':
                rewards[0,k,l] = coverage_reward(x, sensing_rad, cov_entity, cov_var, 3)
                
            elif task == 'navigation':
                rewards[0,k,l] = navigation_reward(x, nav_entity)
            
            
            #r.call_at_scripts_end()

        print('Task: ' + task + ' run: ' + str(l+1) + '/' + str(runs))

    if not (save_path == 'none'):
        with open(save_path + '/reward_' + task  +  '.npy', 'wb') as f:
            np.save(f, rewards)

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly

    print('Done!')

def navigation_reward(state, goals):
    rew = 0
    goals = goals.T
    for l in range(3):
        dists = [np.sqrt(np.sum(np.square(state[:2, :] - goals[:, [i]]))) for i in range(state.shape[1])]
        rew -= min(dists)
    return rew

def transport_reward(state, prev_pay, pay_cap, nec_dis=0.2):
    rew = 0
    for i in range(state.shape[1]):
        if (prev_pay[i] == 1) and (np.linalg.norm(state[:2, i] - np.array([-0.5,0])) < nec_dis):
            rew += pay_cap[i]
        elif (prev_pay[i] == 0) and (np.linalg.norm(state[:2, i] - np.array([0.5,0])) < nec_dis):
            rew += pay_cap[i]
    return rew

def coverage_reward(state,sensing_rad, mus, sigmas, N):
    pos = state[:2, :].reshape((3,1,2))
    res = 100
    x = np.linspace(-1, 1, num=res)
    y = np.linspace(-1, 1, num=res)
    xx = np.ones((res, 1)) * x
    yy = np.ones((1, res)) * y[::-1, np.newaxis]
    grid = np.stack((xx,yy), axis=2)
    cost = np.zeros(xx.shape)
    closest_points = np.zeros((xx.shape[0],xx.shape[1], 2)) + 100
    psi = np.zeros(xx.shape)

    for i in range(2):
        mu_dis = (grid - mus[:, i])
        quad = np.sum(np.dot(mu_dis, np.linalg.inv(np.eye(2) * sigmas[i])) * mu_dis, axis=2)
        psi += 1 / (2 * np.pi * np.sqrt(np.linalg.det(np.eye(2) * sigmas[i]))) * np.exp(- 0.5 * quad)

    for i in range(N):
        dist = np.linalg.norm(grid - pos[i, [0], :], axis=2) ** 2 
        closer_mask = dist < closest_points[...,1] 
        closest_points[closer_mask, 0] = i
        closest_points[closer_mask, 1] = dist[closer_mask]

    for i in range(N):
        dist = np.linalg.norm(grid - pos[i, [0], :], axis=2) ** 2
        voronoi_rad_mask = (closest_points[..., 0] == i) 
        cost[voronoi_rad_mask] += np.power(dist[voronoi_rad_mask], sensing_rad[i]) * psi[voronoi_rad_mask]

    return -np.sum(cost)

if __name__ == "__main__":
    main()
