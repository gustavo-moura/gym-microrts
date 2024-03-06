
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
reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
envs = MicroRTSBotVecEnv(
    ai1s=ai1s,
    ai2s=ai2s,
    max_steps=max_steps,
    render_theme=2,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=reward_weight,
    partial_obs=True,
)
envs.reset()

vulcan = envs.vec_client.botClients[0].ai1  # bot0
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
vulcan.setRewardFunctions(rfs)
vulcan.setRBFDelta(0.15)
vulcan.setRBFEpsilon(1)
vulcan.setSERNActions(5)
vulcan.setSERFactor(10)
vulcan.setRewardWeights(reward_weight)
vulcan.setSelectedRBF(vulcan.RBF_EVAL_BASED)  # RBF original - baseado no eval
# vulcan.setSelectedRBF(vulcan.RBF_REWARDS_BASED)  # RBF nova - baseado no reward

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
    append_bot_results(vulcan, results, infos)
    # if i>=250:
    #     leaf = vulcan.tree.selectLeaf(0, 1, vulcan.epsilon_l, vulcan.epsilon_g, vulcan.epsilon_0, vulcan.global_strategy, vulcan.MAX_TREE_DEPTH, vulcan.current_iteration)
    #     pdb.set_trace()

    if done:
        print_winner(results)
        save_video(images, path=out_path/'experiment.mp4')
        with open(out_path/'results.pkl', 'wb') as file: pkl.dump(results, file)
        #pdb.set_trace()
        break

print('Done.')
envs.close()

