
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder
import matplotlib.pyplot as plt 
import pdb
from tqdm import trange
import pandas as pd

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


# -----------------------------------------------------------------------------
max_steps = 20000


duration = {}

ai1s = [
    microrts_ai.tiamat,
    microrts_ai.coacAI,
    microrts_ai.naiveMCTSAI,
    microrts_ai.vulcanMCTSAI,
]
ai2s = [
    microrts_ai.tiamat,
    microrts_ai.coacAI,
    microrts_ai.naiveMCTSAI,
    microrts_ai.vulcanMCTSAI,
]

winners = np.zeros((len(ai1s), len(ai2s)))
durations = np.zeros((len(ai1s), len(ai2s)))

for one, ai1 in enumerate(ai1s):
    for two, ai2 in enumerate(ai2s):
        print(f"Game {ai1} vs. {ai2}")

        envs = MicroRTSBotVecEnv(
            ai1s=[ai1],
            ai2s=[ai2],
            max_steps=max_steps,
            render_theme=2,
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            shutdown_jvm=False,
        )
        envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x == 0, video_length=max_steps)

        envs.reset()
        bot0 = envs.vec_client.botClients[0].ai1  # Vulcan agent
        bot1 = envs.vec_client.botClients[0].ai2  # NaiveMCTS agent

        import ai.evaluation.SimpleSqrtEvaluationFunction3 as ef
        eval = ef()

        evaluations = []

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

            # Track E()
            gs = envs.vec_client.botClients[0].gs

            e = eval.evaluate(0,1,gs)
            evaluations.append(e)


            # Game over
            if done:
                print("Game over")

                # Winner
                last_eval = evaluations[-2]
                if last_eval > 0:
                    winner = ai1
                    winner_i = one
                    duration[ai2] = len(evaluations)
                elif last_eval < 0:
                    winner = ai2
                    winner_i = two
                    duration[ai1] = len(evaluations)
                else:
                    winner = None

                print("Winner: ", winner)

                winners[one, two] = winner_i
                durations[one, two] = len(evaluations)

                break

        envs.close()


import pdb; pdb.set_trace()

