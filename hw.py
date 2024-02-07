
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
import pdb
from tqdm import trange
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

envs.reset()
bot0 = envs.vec_client.botClients[0].ai1
bot1 = envs.vec_client.botClients[0].ai2

evals_history = []
sers0 = []
risks_history = []
rbfs0 = []

scores_0 = []
scores_1 = []

images = []

for i in trange(max_steps):
    next_obs, reward, done, infos = envs.step([[[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],]])

    img = envs.render(mode='rgb_array')
    images.append(img)
    
    # Get evaluations
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

        pdb.set_trace()

        break

envs.close()


