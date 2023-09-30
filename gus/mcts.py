import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple
import math
import uuid
import copy 

#import graphviz

# a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class WithSnapshots(Wrapper):
    """
    Creates a wrapper that supports saving and loading environemnt states.
    Required for planning algorithms.

    This class will have access to the core environment as self.env, e.g.:
    - self.env.reset()           #reset original env
    - self.env.ale.cloneState()  #make snapshot for atari. load with .restoreState()
    - ...

    You can also use reset() and step() directly for convenience.
    - s = self.reset()                   # same as self.env.reset()
    - s, r, done, _ = self.step(action)  # same as self.env.step(action)
    
    Note that while you may use self.render(), it will spawn a window that cannot be pickled.
    Thus, you will need to call self.close() before pickling will work again.
    """

    def get_snapshot(self, render=False):
        """
        :returns: environment state that can be loaded with load_snapshot 
        Snapshots guarantee same env behaviour each time they are loaded.

        Warning! Snapshots can be arbitrary things (strings, integers, json, tuples)
        Don't count on them being pickle strings when implementing MCTS.

        Developer Note: Make sure the object you return will not be affected by 
        anything that happens to the environment after it's saved.
        You shouldn't, for example, return self.env. 
        In case of doubt, use pickle.dumps or deepcopy.

        """
        if render:
            self.render()  # close popup windows since we can't pickle them
            self.close()
        #print(self.unwrapped)
        # if hasattr(self.unwrapped, 'viewer'):
        #     if self.unwrapped.viewer is not None:
        #         self.unwrapped.viewer.close()
        #         self.unwrapped.viewer = None
    
        #return dumps(self.env)
        gs = self.env.vec_client.clients[0].player1gs.clone()
        return gs

    def load_snapshot(self, snapshot, render=False):
        """
        Loads snapshot as current env state.
        Should not change snapshot inplace (in case of doubt, deepcopy).
        """

        assert not hasattr(self, "_monitor") or hasattr(
            self.env, "_monitor"), "can't backtrack while recording"

        if render:
            self.render()  # close popup windows since we can't load into them
            self.close()
        #self.env = loads(snapshot)
        self.env = snapshot

    def get_result(self, snapshot, action):
        """
        A convenience function that 
        - loads snapshot, 
        - commits action via self.step,
        - and takes snapshot again :)

        :returns: next snapshot, next_observation, reward, is_done, info

        Basically it returns next snapshot and everything that env.step would have returned.
        """
        # <YOUR CODE: load, commit, take snapshot>   
        self.load_snapshot(snapshot)
        observation, reward, done, info = self.env.step(action)        
        next_snapshot = self.get_snapshot()
        
        return ActionResult(
            next_snapshot,
            observation,
            reward,
            done,
            info,
        )


class Node:
    """A tree node for MCTS.
    
    Each Node corresponds to the result of performing a particular action (self.action)
    in a particular state (self.parent), and is essentially one arm in the multi-armed bandit that
    we model in that state."""

    # metadata:
    parent = None  # parent Node
    qvalue_sum = 0.  # sum of Q-values from all visits (numerator)
    visits = 0  # counter of visits (denominator)

    
    def __init__(self, env, parent, action):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.

        :param parent: parent Node
        :param action: action to commit from parent Node
        """
        
        self.id = str(uuid.uuid1())

        self.env = env

        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes

        # get action outcome and save it
        res = env.get_result(parent.snapshot, action)
        self.snapshot, self.observation, self.reward, self.is_done, _ = res

    def __str__(self):
        attrs = vars(self)
        attrs['snapshot'] = 'blob'
        return ', '.join(f"{item[0]}: {item[1]}" for item in attrs.items())
        

    def is_leaf(self):
        return len(self.children) == 0


    def is_root(self):
        return self.parent is None


    def get_qvalue_estimate(self):
        return self.qvalue_sum / self.visits if self.visits != 0 else 0


    def expanded(self):
        return len(self.children) > 0


    def ucb_score(self, scale=10, max_value=1e100):
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From Hoeffding inequality,
                      assumes reward range to be [0, scale].
        :param max_value: a value that represents infinity (for unvisited nodes).

        """

        if self.visits == 0:
            return max_value

        # compute ucb-1 additive component (to be added to mean value)
        # hint: you can use self.parent.visits for N times node was considered,
        # and self.visits for n times it was visited

        N = self.parent.visits
        na = self.visits 
        
        U = math.sqrt((2 * math.log(N)) / na)
        
        Q = self.get_qvalue_estimate() + scale * U       

        return Q


    # MCTS steps
    def select_best_child_new(self):
        if not self.expanded():
            return self.action, self

        # Compute the UCB1 score for each child node
        ucb1 = np.zeros(len(self.children))
        for i, (action, child) in enumerate(self.children.items()):
            q_value = child.reward / child.visits if child.visits > 0 else 0
            exploration_term = exploration_factor * np.sqrt(np.log(self.visits) / child.visits)
            risk_term = np.sqrt(self.parent.visits / (1 + child.visits))

            ucb1[i] = q_value + exploration_term + risk_term

        # Select the child node with the highest UCB1 score
        idx = np.argmax(ucb1)
        action = list(self.children.keys())[idx]
        child = list(self.children.values())[idx]
        return action, child



    def select_best_child(self):
        """
        Picks the child with the highest mean reward
        Chooses between its own child and checks the 
        """
        epsilon = 0.0001
        
        if self.is_leaf():
            return self
        
        best_child = next(iter(self.children))
        for child in self.children:
            if (child.qvalue_sum / (child.visits + epsilon)) > (best_child.qvalue_sum / (best_child.visits + epsilon)):
                best_child = child
                
        return best_child


    def select_best_leaf(self):
        """
        Picks the leaf with the highest priority to expand.
        Does so by recursively picking nodes with the best UCB-1 score until it reaches a leaf.
        """
        if self.is_leaf():
            return self

        # Select the child node with the highest UCB score. You might want to implement some heuristics
        # to break ties in a smart way, although CartPole should work just fine without them.
        best_child = next(iter(self.children))
        for child in self.children:
            if child.ucb_score() > best_child.ucb_score():
                best_child = child
        
        return best_child.select_best_leaf()


    def expand(self):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.
        """

        assert not self.is_done, "can't expand from terminal state"

        for action in range(self.env.action_space.n):
            child = Node(self.env, self, action)
            self.children.add(child)

        # If you have implemented any heuristics in select_best_leaf(), they will be used here.
        # Otherwise, this is equivalent to picking some undefined newly created child node.
        return self.select_best_leaf()


    def rollout(self, t_max=10**4):
        """
        Play the game from this state to the end (done) or for t_max steps.

        On each step, pick action at random (hint: env.action_space.sample()).

        Compute sum of rewards from the current state until the end of the episode.
        Note 1: use env.action_space.sample() for picking a random action.
        Note 2: if the node is terminal (self.is_done is True), just return self.reward.

        """

        # set env into the appropriate state
        self.env.load_snapshot(self.snapshot)
        obs = self.observation
        is_done = self.is_done

        #import pdb; pdb.set_trace()
        
        reward_sum = self.reward

        #<YOUR CODE: perform rollout and compute reward>
        t = 0
        while (not is_done) or t < t_max:
            action = self.env.action_space.sample()
            observation, r, is_done, info = self.env.step(action)
            if is_done:
                self.env.reset()
            reward_sum += r
            t += 1

        if reward_sum > 0:
            r = 1
        else:
            r = 0

        return r
        #return reward_sum


    def propagate(self, child_qvalue):
        """
        Uses child Q-value (sum of rewards) to update parents recursively.
        """
        # compute node Q-value
        my_qvalue = self.reward + child_qvalue

        # update qvalue_sum and visits
        self.qvalue_sum += my_qvalue
        self.visits += 1

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(my_qvalue)


    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child


class Root(Node):
    def __init__(self, env, snapshot, observation):
        """
        creates special node that acts like tree root
        :snapshot: snapshot (from env.get_snapshot) to start planning from
        :observation: last environment observation
        """
        self.env = env
        self.id = str(uuid.uuid1())
        
        self.parent = self.action = None
        self.children = set()  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.reward = 0
        self.is_done = False

    @staticmethod
    def from_node(node):
        """initializes node as root"""
        root = Root(node.env, node.snapshot, node.observation)
        # copy data
        copied_fields = ["qvalue_sum", "visits", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root

def plan_mcts(root, n_iters=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in range(n_iters):
        node = root.select_best_leaf()

        if node.is_done:
            # All rollouts from a terminal node are empty, and thus have 0 reward.
            node.propagate(0)
        else:
            # Expand the best leaf. Perform a rollout from it. Propagate the results upwards.
            # Note that here you have some leeway in choosing where to propagate from.
            # Any reasonable choice should work.
            best_leaf = node.expand()
            reward = best_leaf.rollout(t_max=1)
            best_leaf.propagate(reward)



# Visualize MCTS

def create_dot(root, render=True):
    dot = graphviz.Digraph(comment='The Round Table')
    dot.node(root.id, f'{int(root.qvalue_sum)} / {root.visits}')
    add_childs(dot, root) 

    if render:
        dot.render(directory='doctest-output').replace('\\', '/')

    return dot


def add_childs(dot, parent):
    for child in parent.children:
        if child.visits == 0:
            text = ''
        else:
            text = f'{int(child.qvalue_sum)} / {child.visits}'

        dot.node(child.id, text)
        dot.edge(parent.id, child.id, label=str(child.action))
        add_childs(dot, child)

