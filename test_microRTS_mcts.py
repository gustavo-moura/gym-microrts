import numpy as np
from numpy.random import choice

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
# from stable_baselines3.common.vec_env import VecVideoRecorder

from gus.mcts import WithSnapshots, loads, Root, plan_mcts, create_dot
from tqdm import tqdm


envs = WithSnapshots(MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
))
envs.action_space.seed(0)
root_observation = envs.reset()

import pdb; pdb.set_trace()

root_snapshot = envs.get_snapshot()
root = Root(envs, root_snapshot, root_observation)

# plan from root:
plan_mcts(root, n_iters=100)

total_reward = 0  # sum of rewards
test_env = loads(root_snapshot)  # env used to show progress

acc_rewards = []


for i in tqdm(range(1000)):
    envs.render()
    print(i)
    # TODO: this numpy's `sample` function is very very slow.
    # PyTorch's `sample` function is much much faster,
    # but we want to remove PyTorch as a core dependency...
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_type_mask = action_mask[:, 0:6]
    action = np.concatenate(
        (
            sample(action_mask[:, 0:6]),  # action type
            sample(action_mask[:, 6:10]),  # move parameter
            sample(action_mask[:, 10:14]),  # harvest parameter
            sample(action_mask[:, 14:18]),  # return parameter
            sample(action_mask[:, 18:22]),  # produce_direction parameter
            sample(action_mask[:, 22:29]),  # produce_unit_type parameter
            sample(action_mask[:, 29 : sum(envs.action_space.nvec[1:])]),  # attack_target parameter
        ),
        axis=1,
    )
    action = np.array([envs.action_space.sample()])
    # with np.printoptions(threshold=np.inf):
    #     print(action)
    #     print(f'\n{np.shape(action)}')
    next_obs, reward, done, info = envs.step(action)
    with np.printoptions(threshold=np.inf):
        print(next_obs)
        print(f'\n{np.shape(next_obs)}')
envs.close()
