import plotly
import imageio
import numpy as np

def print_winner(infos, ai1s, ai2s):
    print(f'{infos[0] = }')

    if 'raw_rewards' in infos[0]:
        raw_rewards = infos[0]['raw_rewards']
        if raw_rewards[0] == 1:
            print(f"Winner: {ai1s[0].__name__}")
        elif raw_rewards[0] == -1:
            print(f"Winner: {ai2s[0].__name__}")
        else:
            print("Winner: Draw")


def plotly_sers_rbfs(sers, rbfs, title_a="SER", title_b="RBF", title="SER and RBF"):
    plotly.offline.plot(
        {
            "data": [
                plotly.graph_objs.Scatter(x=list(range(len(sers))), y=sers, name=title_a),
                plotly.graph_objs.Scatter(x=list(range(len(rbfs))), y=rbfs, name=title_b),
            ],
            "layout": plotly.graph_objs.Layout(
                title=title, 
                xaxis={"title": "timestep"},
                yaxis={"title": "value"},
            ),
        }, 
        auto_open=True
    )


def save_video(images, path='videos/test.mp4'):
    writer = imageio.get_writer(path, fps=20)
    for i, img  in enumerate(images):
        if i%2 == 0:
            writer.append_data(np.array(img))
    writer.close()
    print("Video saved")
    return