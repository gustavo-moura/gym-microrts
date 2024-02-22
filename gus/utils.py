import plotly
import imageio
import numpy as np

def winner(results):
    """Return the winner of a game"""
    if 'raw_rewards' in results:
        raw_rewards = results['raw_rewards'][-1]
        if raw_rewards[0] == 1:
            return results['ai1']
        elif raw_rewards[0] == -1:
            return results['ai2']
        else:
            return "Draw"


def print_winner(results):
    """Print the winner of a game"""
    print(f'{results["raw_rewards"][-1] = }')
    won = winner(results)
    print(f'{won = }')


def plotly_sers_rbfs(
        sers, 
        rbfs, 
        title_a="SER", 
        title_b="RBF", 
        title="SER and RBF", 
        mode='lines+markers', 
        save_path='temp-plot.html',
        auto_open=True
    ):
    """Plot two lists of values"""
    plotly.offline.plot(
        {
            "data": [
                plotly.graph_objs.Scatter(x=list(range(len(sers))), y=sers, name=title_a, mode=mode),
                plotly.graph_objs.Scatter(x=list(range(len(rbfs))), y=rbfs, name=title_b, mode=mode),
            ],
            "layout": plotly.graph_objs.Layout(
                title=title, 
                xaxis={"title": "timestep"},
                yaxis={"title": "value"},
            ),
        }, 
        auto_open=True,
        filename=str(save_path),
    )


# plot points
    
def plotly_points(points, title="Points"):
    """Plot a list of points"""
    plotly.offline.plot(
        {
            "data": [
                plotly.graph_objs.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode='markers'),
            ],
            "layout": plotly.graph_objs.Layout(
                title=title, 
                xaxis={"title": "x"},
                yaxis={"title": "y"},
            ),
        }, 
        auto_open=True
    )


def save_video(images, path='videos/test.mp4'):
    """Save a list of images to a video file"""
    writer = imageio.get_writer(path, fps=20)
    for i, img  in enumerate(images):
        if i%2 == 0:
            writer.append_data(np.array(img))
    writer.close()
    print("Video saved")
    return

def numpy_resize(arr):
    """Save a list of numpy arrays to a file, padding them to the same length"""
    counts = [len(a) for a in arr]
    max_e = max(counts)
    np_nan_arr = np.full(max_e, np.nan)
    arr = [np.concatenate([a, np_nan_arr[len(a):]]) for a in arr]
    arr = np.array(arr)
    return arr
    
def save_numpy(arr, path, resize=False):
    """Save a list of numpy arrays to a file
    if resize is True, pad them to the same length
    """
    if resize:
        arr = numpy_resize(arr)
    np.save(path, np.array(arr))


def get_lasts(arr):
    # get the last positive number from each row of arr numpy array
    lasts = []
    for a in arr:
        # get last positive number from the numpy array
        only_nonzero = a[~np.isnan(a)]
        if len(only_nonzero) == 0:
            last = 0
        else:
            last = only_nonzero[-1]
            
        lasts.append(last)
    lasts = np.array(lasts)
    return lasts

def append_bot_results(bot, results, infos):
    results['evals_history'].append(np.array(bot.global_evals))
    results['sers0'].append(np.array(bot.global_ser))
    results['risks_history'].append(np.array(bot.global_risks))
    results['rbfs0'].append(np.array(bot.global_rbf))
    results['scores_0'].append(np.array(bot.global_scores_0))
    results['scores_1'].append(np.array(bot.global_scores_1))
    results['fallbacks'].append(np.array(bot.fallback_actions))
    results['vulcan_rewards'].append(np.array(bot.global_rewards))
    results['raw_rewards'].append(infos[0]['raw_rewards'])
    return results


def treat_fallback_actions(fallbacks):
    
    actions = np.empty(len(fallbacks))
    actions.fill(-1)

    last_len = 0

    for i, fallback_cummulative in enumerate(fallbacks):
        if len(fallback_cummulative) > last_len:
            actions[i] = fallback_cummulative[-1]
            last_len = len(fallback_cummulative)

    return actions
