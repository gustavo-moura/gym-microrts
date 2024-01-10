
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
from stable_baselines3.common.vec_env import VecMonitor
import matplotlib.pyplot as plt 
import pdb
import plotly
from tqdm import trange

max_steps = 1000
envs = MicroRTSBotVecEnv(
    ai1s=[microrts_ai.vulcanMCTSAI],
    ai2s=[microrts_ai.naiveMCTSAI],
    max_steps=max_steps,
    render_theme=2,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    #map_paths=["maps/12x12/complexBasesWorkers12x12.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
envs.reset()
bot0 = envs.vec_client.botClients[0].ai1  # Vulcan agent
bot1 = envs.vec_client.botClients[0].ai2  # Vulcan agent



import ai.evaluation.SimpleSqrtEvaluationFunction3 as ef
eval = ef()


sers0 = []
sers1 = []

rbfs0 = []
rbfs1 = []

# List of many state history's evaluations
evals_history = [] # for the simulated state history

def plot_evals(evals):
    # evals for a state history
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('evaluations')
    ax.set_ylabel('E()')
    ax.set_xlabel('timestep')
    ax.plot(evals, label='evaluations(0,1)')
    ax.legend()
    plt.show()


primeiro = True

for i in trange(max_steps):
    envs.render()
    next_obs, reward, done, infos = envs.step(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ]
    )

    # envs.vec_client.botClients[0].ai1
    
    # global_evals = envs.vec_client.botClients[0].ai1.global_evals
    # envs.vec_client.botClients[0].ai1.sequence_execution_risk(global_evals)
    
    # For the simulated state history, get current SER
    #bot = envs.vec_client.botClients[0].ai1  # Vulcan agent
    sers0.append(bot0.global_ser)
    evals_history.append(bot0.global_evals)
    rbfs0.append(bot0.global_rbf)

    #sers1.append(bot1.global_ser)  #23/11
    #rbfs1.append(bot1.global_rbf)  #23/11

    # Track E()
    #gs = envs.vec_client.botClients[0].gs

    # e0 = eval.evaluate(0,1,gs)
    # evaluations0.append(e0)

    # e1 = eval.evaluate(1,0,gs)
    # evaluations1.append(e1)

    #pdb.set_trace()

    # Game over
    if done:
        # plot the evaluations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('')
        ax.set_ylabel('value')
        ax.set_xlabel('timestep')
        ax.plot(sers0, label='SER Bot0')
        #ax.plot(sers1, label='SER Bot1')
        ax.plot(rbfs0, label='RBF Bot0')
        #ax.plot(rbfs1, label='RBF Bot1')
        #ax.plot(rbfs0, label='RBF (ser_reward)')
        # ax.plot(evaluations1, label='evaluations(1,0)')
        ax.legend()
        plt.show()

        # plotly.offline.plot({
        #     "data": [
        #         plotly.graph_objs.Scatter(x=list(range(len(sers0))), y=sers0, name="SER Bot0"),
        #         plotly.graph_objs.Scatter(x=list(range(len(rbfs0))), y=rbfs0, name="RBF Bot0"),
        #     ],
        #     "layout": plotly.graph_objs.Layout(
        #         title="SER and RBF", 
        #         xaxis={"title": "timestep"},
        #         yaxis={"title": "value"},
        #     ),
        # }, auto_open=True)


        pdb.set_trace()
        break




envs.close()