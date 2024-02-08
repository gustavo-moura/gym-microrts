
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
import pdb
from tqdm import trange
from gus.utils import print_winner, plotly_sers_rbfs, save_video, save_numpy_resize, save_numpy
from pathlib import Path
from jpype.types import JArray
import pickle as pkl

ai1s = [microrts_ai.vulcanMCTSAI]
#ai2s = [microrts_ai.naiveMCTSAI]
#ai2s = [microrts_ai.coacAI]
ai2s = [microrts_ai.lightRushAI]

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

# setup the bot0
bot0 = envs.vec_client.botClients[0].ai1
from ai.rewardfunction import (
    AttackRewardFunction,
    ProduceBuildingRewardFunction,
    ProduceCombatUnitRewardFunction,
    ProduceWorkerRewardFunction,
    ResourceGatherRewardFunction,
    RewardFunctionInterface,
    WinLossRewardFunction,
)

rfs = JArray(RewardFunctionInterface)(
    [
        WinLossRewardFunction(),
        ResourceGatherRewardFunction(),
        ProduceWorkerRewardFunction(),
        ProduceBuildingRewardFunction(),
        AttackRewardFunction(),
        ProduceCombatUnitRewardFunction()
    ]
)
bot0.setRewardFunctions(rfs)
# bot0.setRBFDelta(0.01)


evals_history = []
sers0 = []
risks_history = []
rbfs0 = []

scores_0 = []
scores_1 = []

images = []

raw_rewards = []

vulcan_rewards = []

fallbacks = []

for i in trange(max_steps):
    try:
        next_obs, reward, done, infos = envs.step([[[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],]])
    except Exception as e:
        print(e)
        pdb.set_trace()

    raw_rewards.append(infos[0]['raw_rewards'])

    img = envs.render(mode='rgb_array')
    images.append(img)
    
    # Get evaluations
    evals_history.append(bot0.global_evals)
    sers0.append(bot0.global_ser)

    risks_history.append(bot0.global_risks)
    rbfs0.append(bot0.global_rbf)

    scores_0.append(bot0.global_scores_0)
    scores_1.append(bot0.global_scores_1)

    vulcan_rewards.append(np.array(bot0.global_rewards))

    fallbacks.append(bot0.fallback_actions)

    if done:
        print_winner(infos, ai1s, ai2s)

        out_path = Path('./videos/experiment')
        out_path.mkdir(parents=True, exist_ok=True)

        save_video(images, path=out_path/'experiment.mp4')

        plotly_sers_rbfs(sers0, rbfs0, save_path=out_path/'plot_sers_rbfs.html')
        save_numpy_resize(evals_history, out_path/'evals_history.npy')
        save_numpy_resize(risks_history, out_path/'risks_history.npy')
        save_numpy_resize(scores_0, out_path/'scores_0.npy')
        save_numpy_resize(scores_1, out_path/'scores_1.npy')

        save_numpy(raw_rewards, out_path/'raw_rewards.npy')
        with open(out_path/'vulcan_rewards.pkl', 'wb') as file: pkl.dump(vulcan_rewards, file)

        pdb.set_trace()
        

        break

envs.close()


