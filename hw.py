
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
import pdb
from tqdm import trange
from gus.utils import print_winner, save_video, append_bot_results
from pathlib import Path
from jpype.types import JArray
import pickle as pkl

out_path = Path('./videos/experiment')
out_path.mkdir(parents=True, exist_ok=True)

ai1s = [microrts_ai.vulcanMCTSAI]
ai2s = [microrts_ai.naiveMCTSAI]
#ai2s = [microrts_ai.coacAI]
#ai2s = [microrts_ai.lightRushAI]

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
bot0.setRBFDelta(0.15)

images = []

results = {
    'ai1': ai1s[0].__name__,
    'ai2': ai2s[0].__name__,
    'evals_history': [],
    'sers0': [],
    'risks_history': [],
    'rbfs0': [],
    'scores_0': [],
    'scores_1': [],
    'fallbacks': [],
    'vulcan_rewards': [],
    'raw_rewards': [],
}


for i in trange(max_steps):
    next_obs, reward, done, infos = envs.step([[[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],]])

    img = envs.render(mode='rgb_array')
    images.append(img)

    # Get evaluations
    append_bot_results(bot0, results, infos)


    if done:
        print_winner(results)
        save_video(images, path=out_path/'experiment.mp4')
        with open(out_path/'results.pkl', 'wb') as file: pkl.dump(results, file)

        #print(f'Results fallback actions: {results["fallbacks"]}')

        #pdb.set_trace()

        break

print('Done.')

envs.close()

