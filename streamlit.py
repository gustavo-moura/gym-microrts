from pathlib import Path
import streamlit as st
from gus.utils import numpy_resize, winner, get_lasts
import pickle as pkl
import numpy as np
import plotly.graph_objs as go

# Setup
colors1 = [
    'rgb(90, 98, 242)',  # blue
    'rgb(215, 86, 60)',  # red
    'rgb(85, 194, 142)',  # green
]
colors2 = [
    '#003f5c',
    '#2f4b7c',
    '#665191',
    '#a05195',
    '#d45087',
    '#f95d6a',
    '#ff7c43',
    '#ffa600',
]

out_path = Path('./videos/experiment')
r = pkl.load(open(out_path / 'results.pkl', 'rb'))


# Plot 1

st.title(f'{r["ai1"]} vs. {r["ai2"]}')
st.video(str(out_path / 'experiment.mp4'))


st.title(f'Winner: {winner(r)}')


# Plot 2 - RBFs, SERs and E()
evals = numpy_resize(r['evals_history'])
lasts_evals = get_lasts(evals)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(r['sers0'])), y=r['sers0'], name='SER - risco', mode='lines+markers', line=dict(color=colors1[0])))
fig.add_trace(go.Scatter(x=np.arange(len(r['rbfs0'])), y=r['rbfs0'], name='RBF - limitante de risco', mode='lines+markers', line=dict(color=colors1[1])))
fig.add_trace(go.Scatter(x=np.arange(len(lasts_evals)), y=lasts_evals, name='E() - eval', mode='lines+markers', line=dict(color=colors1[2])))
fig.update_layout(title='SER and RBF', xaxis_title='timestep', yaxis_title='value')
st.plotly_chart(fig, use_container_width=True)


# Plot 3 - Fallback actions

# plot if there are fallbacks in a binary way
fallbacks = numpy_resize(r['fallbacks'])
fallbacks[0:20] = np.array([0,1,1,1,0,1,1,0,0])
fallbacks = fallbacks.astype(int)
st.text(fallbacks)

# for each fallback, plot a "ON/OFF Filling" bar that is created by using two complementary logic lines (that is, one line is always the reverse of the other line).
fig = go.Figure()
for i, f in enumerate(fallbacks):
    fig.add_trace(go.Scatter(x=np.arange(len(f)), y=f, mode='lines', fill='tonexty', line=dict(color='green', width=2, shape='hv')))
    # inverse f, fill the gaps
    fig.add_trace(go.Scatter(x=np.arange(len(f)), y=1-f, mode='lines', line=dict(color='red', width=2, shape='hv')))
    break
fig.update_layout(title='Fallback actions', xaxis_title='timestep', yaxis_title='fallback')
st.plotly_chart(fig, use_container_width=True)





