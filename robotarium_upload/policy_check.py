#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from reward_net import RewardNet



# Some gains for this experiment.  These aren't incredibly relevant.
d = 0.3
ddiag = np.sqrt(5)*d
formation_control_gain = 10

# Experiment constants
iterations = 2000
dt = 1
N = 3

#Limit maximum linear speed of any robot
magnitude_limit = 10

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()
dxi = np.zeros((2, N))

policy = [RewardNet(policy_path='model.npy', model_choice=i) for i in range(N)]

obs_size = 9 
trajs = np.zeros((3, obs_size, iterations))

prev_pay = np.ones((3,))

for k in range(iterations):

    # Get the poses of the robots
    x = r.get_poses()

    # Initialize a velocity vector
    F = np.zeros((2,N))

    nav_entity = np.array([[1,0,-1], [0,0,0]]).T
    cov_entity = np.array([[-1,1], [0,0]])
    cov_var = np.array([1,1])

    print(x[0, :])
    
    for i in range(N):
        # Navigation Observation Space
        # obs = np.zeros((14,))
        # obs[:2] = np.squeeze(dxi[:, i])
        # obs[2:4] = np.squeeze(x[:2, i])
        # obs[4:10] = (nav_entity - x[:2, [i]].T).flatten()
        # obs[10:] = (x[:2, [l for l in range(N) if not (l == i)]] - x[:2,i]).flatten()

        # Transport Observation Space
        obs = np.zeros((9,))
        obs[:2] = np.squeeze(dxi[:, i])
        obs[2:4] = np.squeeze(x[:2, i])
        if (prev_pay[i] == 1) and (np.linalg.norm(x[:2, i] - np.array([1, 0])) < 0.4):
            prev_pay[i] = 0
            
        print(prev_pay)
        if (prev_pay[i] == 0) and (np.linalg.norm(x[:2, i] - np.array([-1, 0])) < 0.4):
            prev_pay[i] = 1
            
        obs[4] = 1-prev_pay[i]
        obs[5:] = np.array([1, 0, -1, 0])




        # Coverage Observation Space
        # obs = np.zeros((14,))
        # cov1 = cov_entity[:,0]
        # cov2 = cov_entity[:,1]
        # var1 = cov_var[0]
        # var2 = cov_var[1]
        # obs[:2] = cov1 - x[:2, i]
        # obs[2] = var1
        # obs[3:5] = cov2 - x[:2, i]
        # obs[5] = var2
        # obs[6:10] = (x[:2, [l for l in range(N) if not (l == i)]] - x[:2,i]).flatten()
        # obs[10:12] = np.squeeze(x[:2, i])
        # obs[12:] = np.squeeze(dxi[:, i])

        trajs[i, :, k] = obs
        act = np.argmax(policy[i].get_action(obs))

        if act == 1:
            F[0, i] = 1
        elif act == 2:
            F[0, i] = -1
        elif act == 3:
            F[1, i] = 1
        elif act == 4:
            F[1, i] = -1

    
    dxi += F * dt


    #Keep single integrator control vectors under specified magnitude
    # Threshold control inputs
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    # Make sure that the robots don't collide
    dxi = si_barrier_cert(dxi, x[:2, :])

    # Transform the single-integrator dynamcis to unicycle dynamics
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities of the robots
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

with open('trajectories.npy', 'wb') as f:
    np.save(f, trajs)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
