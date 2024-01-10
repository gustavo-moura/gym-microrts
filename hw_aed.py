
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import numpy as np
from stable_baselines3.common.vec_env import VecMonitor
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


# Metrics for calculating the quality of a game design
# Browne 2011, Evolutionary Game Design

class Metrics:

    def __init__(self):
        pass

    def drama(self, evaluations, winner):
        """Calculates the drama of a game, given the evaluations and the winner.
        Drama is how much the loser was in advantage during the game.

        Args:
            evaluations (list): list of evaluations of a game.
            winner (int): player who won the game, 0 or 1.
                P0: maximize (blue)
                P1: minimize (red)
        """

        drama = 0

        if winner == 0: # maximize
            for e in evaluations:
                if e < 0:  # Sums how much the loser was in advantage
                    drama += abs(e)
        
        elif winner == 1: # minimize
            for e in evaluations:
                if e > 0:
                    drama += e
        
        drama = drama / len(evaluations)

        return drama
        
    def uncertainty(self, evaluations, winner):
        """Calculates the uncertainty of a game, given the evaluations and the winner.
        Uncertainty uncertainty is the difference between the linear evaluation 
        and the real evaluation of the winner player.

        Args:
            evaluations (list): list of evaluations of a game.
            winner (int): player who won the game, 0 or 1.
                P0: maximize (blue)
                P1: minimize (red)
        """

        n = len(evaluations)

        uncertainty = 0

        # linear (boring) evaluation expected of the winner player
        # linear ascending for P0, linear descending for P1
        if winner == 0:
            linear_eval = np.linspace(0, 1, n)
            uncertainty = np.sum(linear_eval - evaluations) / n

        elif winner == 1:
            linear_eval = np.linspace(0, -1, n)
            uncertainty = np.sum(evaluations - linear_eval) / n

        return uncertainty
                
    def lead_change(self, evaluations):
        """Calculates the lead change of a game, given the evaluations.
        Lead change is how many times the leader changed.
        
        Args:
            evaluations (list): list of evaluations of a game.
        """

        lead_change = 0
        n = len(evaluations)

        # Count how many time the evaluations changed sign
        for i in range(n - 1):
            if evaluations[i] * evaluations[i+1] < 0:
                lead_change += 1

        lead_change = lead_change / (n - 1)

        return lead_change


# -----------------------------------------------------------------------------
max_steps = 3000

metrics = Metrics()


metrics_results = []

for map_i in range(10):
    print("Game", map_i)
    map_name = f"maps/AED/round_1_map_{map_i:03d}.xml"

    envs = MicroRTSBotVecEnv(
        ai1s=[microrts_ai.vulcanMCTSAI],
        #ai2s=[microrts_ai.naiveMCTSAI],
        ai2s=[microrts_ai.tiamat],
        max_steps=max_steps,
        render_theme=2,
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        shutdown_jvm=False,
    )
    envs.reset()
    bot0 = envs.vec_client.botClients[0].ai1  # Vulcan agent
    bot1 = envs.vec_client.botClients[0].ai2  # NaiveMCTS agent

    import ai.evaluation.SimpleSqrtEvaluationFunction3 as ef
    eval = ef()

    evaluations = []

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
            winner = 0
        elif last_eval < 0:
            winner = 1
        else:
            winner = None

        print("Winner: P", winner)
        #print("Evaluations: ", evaluations)
        #plot_evals(evaluations)

        # Metrics
        drama = metrics.drama(evaluations, winner)
        uncertainty = metrics.uncertainty(evaluations, winner)
        lead_change = metrics.lead_change(evaluations)

        print("Drama:       ", drama)
        print("Uncertainty: ", uncertainty)
        print("Lead change: ", lead_change)

        metrics_results.append({
            "map": map_name,
            "drama": drama,
            "uncertainty": uncertainty,
            "lead_change": lead_change,
        })

    envs.close()

    # import time
    # time.sleep(10)

metrics_df = pd.DataFrame(metrics_results)
print(metrics_df)
metrics_df.to_csv("metrics_results_round_1.csv")