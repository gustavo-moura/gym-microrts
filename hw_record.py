
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder
import matplotlib.pyplot as plt 
import pdb
import plotly
from tqdm import trange
import imageio


max_steps = 2000
envs = MicroRTSBotVecEnv(
    ai1s=[microrts_ai.vulcanMCTSAI],
    ai2s=[microrts_ai.naiveMCTSAI],
    max_steps=max_steps,
    render_theme=2,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
envs = VecVideoRecorder(
    envs, 
    'videos', 
    record_video_trigger=lambda x: x == 0, 
    video_length=max_steps,
    name_prefix=f"experiment"
)

envs.reset()
bot0 = envs.vec_client.botClients[0].ai1
bot1 = envs.vec_client.botClients[0].ai2


import ai.evaluation.SimpleSqrtEvaluationFunction3 as ef
eval = ef()


sers0 = []
sers1 = []

rbfs0 = []
rbfs1 = []

# List of many state history's evaluations
evals_history = [] # for the simulated state history
risks_history = [] # for the simulated state history

# images = []

for i in trange(max_steps):
    #envs.render()
    next_obs, reward, done, infos = envs.step(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ]
    )
    # img = envs.render(mode='rgb_array')
    # images.append(img)
    
    # For the simulated state history, get current SER
    sers0.append(bot0.global_ser)
    evals_history.append(bot0.global_evals)
    risks_history.append(bot0.global_risks)
    rbfs0.append(bot0.global_rbf)

    if done:

        plotly.offline.plot({
            "data": [
                plotly.graph_objs.Scatter(x=list(range(len(sers0))), y=sers0, name="SER Bot0"),
                plotly.graph_objs.Scatter(x=list(range(len(rbfs0))), y=rbfs0, name="RBF Bot0"),
            ],
            "layout": plotly.graph_objs.Layout(
                title="SER and RBF", 
                xaxis={"title": "timestep"},
                yaxis={"title": "value"},
            ),
        }, auto_open=True)


        # imageio.mimsave(f'videos/experiment.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

        pdb.set_trace()
        break

envs.close()

#evals_history_np = np.array(evals_history)
#(Pdb) np.save('evals_history', evals_history_np)

#risks_history_np = np.array(risks_history)
#(Pdb) np.save('risks_history', risks_history_np)