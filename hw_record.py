
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder
import matplotlib.pyplot as plt 
import pdb
from tqdm import trange
import imageio
from gus.utils import print_winner, plotly_sers_rbfs, save_video
from pathlib import Path

ai1s = [microrts_ai.vulcanMCTSAI]
ai2s = [microrts_ai.naiveMCTSAI]

max_steps = 2000
envs = MicroRTSBotVecEnv(
    ai1s=ai1s,
    ai2s=ai2s,
    max_steps=max_steps,
    render_theme=2,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(
#     envs, 
#     'videos', 
#     record_video_trigger=lambda x: x == 0, 
#     video_length=max_steps+1,
#     name_prefix=f"experiment"
# )

envs.reset()
bot0 = envs.vec_client.botClients[0].ai1

# teste
bot0.setMaxTreeDepth(100)

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

scores_0 = []
scores_1 = []

images = []

for i in trange(max_steps):
    #envs.render(mode='rgb_array')
    next_obs, reward, done, infos = envs.step([[[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],]])

    img = envs.render(mode='rgb_array')
    images.append(img)
    
    # For the simulated state history, get current SER
    evals_history.append(bot0.global_evals)
    sers0.append(bot0.global_ser)
    risks_history.append(bot0.global_risks)
    rbfs0.append(bot0.global_rbf)

    scores_0.append(bot0.global_scores_0)
    scores_1.append(bot0.global_scores_1)

    if done:
        print_winner(infos, ai1s, ai2s)
        plotly_sers_rbfs(sers0, rbfs0)

        out_path = Path('./videos/experiment')
        out_path.mkdir(parents=True, exist_ok=True)

        save_video(images, path=out_path/'experiment.mp4')

        np.save(out_path/'evals_history', np.array(evals_history))
        np.save(out_path/'risks_history', np.array(risks_history))
        np.save(out_path/'scores_0', np.array(scores_0))
        np.save(out_path/'scores_1', np.array(scores_1))

        #plotly_sers_rbfs(scores_0, scores_1, title_a="Scores 0", title_b="Scores 1", title="Scores 0 and 1")

        #imageio.mimsave(f'videos/experiment.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

        break

envs.close()


