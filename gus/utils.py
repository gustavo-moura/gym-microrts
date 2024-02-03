import plotly
import imageio
import numpy as np

def print_winner(infos, ai1s, ai2s):
    if infos[0] == 1:
        print(f"Winner: {ai1s[0].__name__}")
    elif infos[0] == -1:
        print(f"Winner: {ai2s[0].__name__}")
    else:
        print("Winner: Draw")


def plotly_sers_rbfs(sers, rbfs, title_suffix=""):
    plotly.offline.plot(
        {
            "data": [
                plotly.graph_objs.Scatter(x=list(range(len(sers))), y=sers, name=f"SER{title_suffix}"),
                plotly.graph_objs.Scatter(x=list(range(len(rbfs))), y=rbfs, name=f"RBF{title_suffix}"),
            ],
            "layout": plotly.graph_objs.Layout(
                title="SER and RBF", 
                xaxis={"title": "timestep"},
                yaxis={"title": "value"},
            ),
        }, 
        auto_open=True
    )


def save_video(images):
    writer = imageio.get_writer('videos/test.mp4', fps=20)
    for i, img  in enumerate(images):
        if i%2 == 0:
            writer.append_data(np.array(img))
    writer.close()
    print("Video saved")
    return