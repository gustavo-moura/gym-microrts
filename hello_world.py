import numpy as np
from numpy.random import choice

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
#from stable_baselines3.common.vec_env import VecVideoRecorder



envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    #ai2s=[microrts_ai.coacAI for _ in range(1)],
    ai2s=[microrts_ai.vulcanMCTSAI for _ in range(1)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
#envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

# bultin_ai vs builtin_ai
# MicroRTSBotVecEnv(
#     ai1s=built_in_ais,
#     ai2s=built_in_ais2,
#     max_steps=max_steps,
#     render_theme=2,
#     map_paths=[map_path],
#     reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
# )

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0:
        return 0
    return choice(range(len(logits)), p=logits / sum(logits))


envs.action_space.seed(0)
envs.reset()
print(envs.action_plane_space.nvec)
nvec = envs.action_space.nvec


def sample(logits):
    return np.array([choice(range(len(item)), p=softmax(item)) for item in logits]).reshape(-1, 1)

import ai.evaluation.SimpleSqrtEvaluationFunction3 as ef
eval = ef()


evaluations0 = []
evaluations1 = []


# Import MCTS agent VULCAN
# from rts.units import UnitTypeTable()
# real_utt = UnitTypeTable()
# vulcan_ai = microrts_ai.vulcanMCTSAI(real_utt)


for i in range(1000):
    #envs.render()
    #print(i)
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
    #action = np.array([envs.action_space.sample()])
    # with np.printoptions(threshold=np.inf):
    #     print(action)
    #     print(f'\n{np.shape(action)}')
    import pdb; pdb.set_trace()
    next_obs, reward, done, info = envs.step(action)

    #with np.printoptions(threshold=np.inf):
        #print(next_obs)
        #print(f'\n{np.shape(next_obs)}')

    if i%100 == 0:
        gs = envs.vec_client.clients[0].gs
        e0 = eval.evaluate(0,1,gs)
        e1 = eval.evaluate(1,0,gs)
        #print(f"{e = }")
        evaluations0.append(e0)
        evaluations1.append(e1)

        # VULCAN
        # vulcan_ai.getAction(0, gs)


# plot the evaluations
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('evaluations')
ax.set_ylabel('E()')
ax.set_xlabel('timestep')
ax.plot(evaluations0, label='evaluations(0,1)')
ax.plot(evaluations1, label='evaluations(1,0)')
ax.legend()
plt.show()



# import pdb; pdb.set_trace()


envs.close()
