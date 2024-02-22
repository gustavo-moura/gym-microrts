from pathlib import Path
import streamlit as st
import gus.utils as utils
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
colors3 = {
    'pink': 'rgb(165, 99, 127)',
    'yellow': 'rgb(234, 249, 116)',
    'darkblue': 'rgb(12, 34, 49)',
}

pathlist = [p for p in Path('./videos').glob('*') if p.is_dir()]
out_path = st.selectbox('Select experiment', pathlist)
r = pkl.load(open(out_path / 'results.pkl', 'rb'))


# Plot 1 - VIDEO REPLAY
st.header(f'{r["ai1"]} vs. {r["ai2"]}')
st.video(str(out_path / 'experiment.mp4'))
st.markdown(f'### Winner: {utils.winner(r)}')


# Plot 2 - RBFs, SERs and E()
st.header('SER and RBF')
evals = utils.numpy_resize(r['evals_history'])
lasts_evals = utils.get_lasts(evals)
linestyle = st.selectbox('Line style', ['lines', 'lines+markers', 'markers'], label_visibility='collapsed')
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(r['sers0'])), y=r['sers0'], name='SER - risco', mode=linestyle, line=dict(color=colors1[0])))
fig.add_trace(go.Scatter(x=np.arange(len(r['rbfs0'])), y=r['rbfs0'], name='RBF - limitante de risco', mode=linestyle, line=dict(color=colors1[1])))
fig.add_trace(go.Scatter(x=np.arange(len(lasts_evals)), y=lasts_evals, name='E() - eval', mode=linestyle, line=dict(color=colors1[2])))
fig.update_layout(xaxis_title='timestep', yaxis_title='value', legend=dict(yanchor="bottom", y=1, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)


# Plot 3 - Actions
st.header('Actions')
fallbacks = utils.numpy_resize(r['fallbacks'])
actions = utils.treat_fallback_actions(r['fallbacks'])
reshape = st.number_input('Divide in lines', min_value=1, max_value=10, value=1, step=9, key='actions')
actions = actions.reshape(reshape, actions.shape[0]//reshape)
st.text(f'{actions.shape = }')
fig = go.Figure()
fig.add_trace(go.Heatmap(z=actions, colorscale='thermal', showscale=False, xaxis='x', yaxis='y', xgap=0))
fig.update_layout(xaxis_title='timestep', yaxis_showticklabels=False)
st.plotly_chart(fig, use_container_width=True)


# PLOT 4 - Pie plot actions ratio
fig = go.Figure()
fig.add_trace(go.Pie(labels=['No Action', 'AI selected action', 'Fallback action'], values=[np.sum(actions == -1), np.sum(actions == 0), np.sum(actions == 1)], hole=.3, marker=dict(colors=[colors3['darkblue'], colors3['pink'], colors3['yellow']])))
st.plotly_chart(fig, use_container_width=True)


rewards = r['vulcan_rewards']
# official timestep
# simulated timestep
# reward functions

reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])

def reward_function(rewards, reward_weight):
    weighted = rewards * reward_weight
    return weighted.sum()

def calculate_rewards(rewards, reward_weight):
    s = np.apply_along_axis(reward_function, 0, rewards, reward_weight)
    gamma = 0.99
    sum_rw = 0
    for t, reward in enumerate(s):
        sum_rw += reward * (gamma**t)
    return sum_rw

all_rewards = []
for reward_in_timestep in rewards:
    current_reward = calculate_rewards(reward_in_timestep, reward_weight)
    all_rewards.append(current_reward)
all_rewards = np.array(all_rewards)

# PLOT 5 - Rewards
st.header('Rewards')
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(all_rewards)), y=all_rewards, name='Vulcan rewards', mode='lines+markers', line=dict(color=colors1[0])))
fig.update_layout(xaxis_title='timestep', yaxis_title='reward', legend=dict(yanchor="bottom", y=1, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)


delta = 0.15
def rbf(rewards):
    return rewards * delta

risks = rbf(all_rewards)

# PLOT 6 - Risks
st.header('RBF - Função Limitante de Risco')
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(risks)), y=risks, name='Risks', mode=linestyle, line=dict(color=colors1[1])))
fig.update_layout(xaxis_title='timestep', yaxis_title='risk', legend=dict(yanchor="bottom", y=1, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)


# Plot 2 - RBFs, SERs and E()
st.header('SER and RBF')
fig = go.Figure()
epsilon = 0.1
fig.add_trace(go.Scatter(x=np.arange(len(r['sers0'])), y=r['sers0'], name='SER - risco: ser(E())', mode=linestyle, line=dict(color=colors1[0])))
fig.add_trace(go.Scatter(x=np.arange(len(r['rbfs0'])), y=r['rbfs0'], name='RBF1 - limitante de risco: rbf(E())', mode=linestyle, line=dict(color=colors1[1])))
fig.add_trace(go.Scatter(x=np.arange(len(lasts_evals)), y=lasts_evals, name='E() - eval', mode=linestyle, line=dict(color=colors1[2])))
fig.add_trace(go.Scatter(x=np.arange(len(risks)), y=risks, name=f'RBF2 - limitante de risco: rbf(rewards)  -> rbf(delta * slc), delta={delta}', mode=linestyle, line=dict(color=colors1[1])))
fig.add_trace(go.Scatter(x=np.arange(len(all_rewards)), y=all_rewards*epsilon, name=f'rewards*{epsilon} - expected weighted rewards', mode=linestyle, line=dict(color=colors1[2])))

fig.update_layout(xaxis_title='timestep', yaxis_title='value', legend=dict(yanchor="bottom", y=1, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)

# ''' Plot EXAMPLE : binary lines
# fallbacks[0:5] = np.array([0,1,1,1,0,1,1,0,0])
# fallbacks[5:10] = np.array([1,1,0,0,0,0,1,0,1])
# st.text(f'Fallback actions: {fallbacks}')
# fallbacks = fallbacks.astype(int)
# fig = go.Figure()
# for i, f in enumerate(fallbacks):
#     fig.add_trace(go.Scatter(x=np.arange(len(f)), y=1-f, name='AI selected action', mode='lines', fill='tonexty', line=dict(color='green', width=0, shape='hv')))
#     fig.add_trace(go.Scatter(x=np.arange(len(f)), y=f, name='Fallback action', mode='lines', fill='tozerox',  line=dict(color='red', width=0, shape='hv')))
#     break
# fig.update_layout(title='Fallback actions', xaxis_title='timestep', yaxis_title='fallback', xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=True, tickson="boundaries", ticklabelposition='inside right'))
# st.plotly_chart(fig, use_container_width=True)
# '''

# ''' PLOT EXAMPLE : heatmap
# fig = go.Figure()
# fig.add_trace(go.Heatmap(
#     z=fallbacks.T,
#     colorscale=[[0, 'grey'], [1, 'red']],
#     showscale=False,
#     xaxis='x',
#     yaxis='y',
#     xgap=0,
# ))
# fig.update_layout(title='Fallback actions', xaxis_title='official timestep', yaxis_title='simulated timestep')
# st.plotly_chart(fig, use_container_width=True)
# st.text('RED  [1]: Fallback action\nGRAY [0]: AI selected action')
# '''
