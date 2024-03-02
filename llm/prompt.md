

Given the current state of MicroRTS, including information about your bases, units, and resource collection, please provide guidance on the next best action to execute, considering factors such as unit production, resource management, and strategic positioning.

Consider those informations for the next interactions.




1. You can only use the following functions. Don’t make plans purely based on your experience, think about how to use these functions.

explore(object, strategy)
Move around to find the object with the strategy: used to find objects including block items and entities. This action is finished once the object is visible (maybe at the distance). Augments:
- object: a string, the object to explore.
- strategy: a string, the strategy for exploration.


2. You cannot define any new function. Note that the "Generated structures" world creation option is turned off.

3. There is an inventory that stores all the objects I have. It is not an entity, but objects can be added to it or retrieved from it anytime at anywhere without specific actions. The mined or crafted objects will be added to this inventory, and the materials and tools to use are also from this inventory. Objects in the inventory can be directly used. Don’t write the code to obtain them. If you plan to use some object not in the inventory, you should first plan to obtain it. You can view the inventory as one of my states, and it is written in form of a dictionary whose keys are the name of the objects I have and the values are their quantities.

4. You will get the following information about my current state:
- inventory: a dict representing the inventory mentioned above, whose keys are the name of
the objects and the values are their quantities
- environment: a string including my surrounding biome, the y-level of my current location,
and whether I am on the ground or underground
Pay attention to this information. Choose the easiest way to achieve the goal conditioned on my current state. Do not provide options, always make the final decision.

5. You must describe your thoughts on the plan in natural language at the beginning. After
that, you should write all the actions together. The response should follow the format: {
"explanation": "explain why the last action failed, set to null for the first planning", "thoughts": "Your thoughts on the plan in natural languag",
"action_list": [
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"},
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"},
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"}
] }
The action_list can contain arbitrary number of actions. The args of each action should
correspond to the type mentioned in the Arguments part. Remember to add “‘dict“‘ at the beginning and the end of the dict. Ensure that you response can be parsed by Python json.loads

6. I will execute your code step by step and give you feedback. If some action fails, I will stop at that action and will not execute its following actions. The feedback will include error messages about the failed action. At that time, you should replan and write the new code just starting from that failed action.




========================================

OBSERVATION

- Resources: A1 (10 remaining)

**Player:**

- Bases: B2
- Workers: A3 (idle)
- Barracks: Absent
- Attack Units: Absent

**Enemy:**

- Bases: D4 (producing worker on C4)
- Workers: Absent
- Barracks: Absent
- Attack Units: Absent



Provide the next set of micro immediate actions, following the structure:

(tile, action, direction, produce_unit_type::optional)

Where:

tile:
- rows A to D
- columns 1 to 4

actions:
- noop
- move
- harvest
- return
- produce
- attack

direction:
- north
- east
- south
- west

produce_unit_type:
- resource
- base
- barracks
- worker
- light
- heavy
- ranged

Example:
(A1, move, north)
(C4, produce, south, worker)

5. You must describe your thoughts on the plan in natural language at the beginning. After
that, you should write all the actions together. 


RESPONSE FORMAT:

The response should follow the format: 
{
"explanation": "explain why the last action failed, set to null for the first planning", 
"thoughts": "Your thoughts on the plan in natural languag",
"action_list": [
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"},
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"},
{"name": "action name", "args": {"arg name": value}, "expectation": "describe the expected results of this action"}
] 
}

The action_list can contain arbitrary number of actions. The args of each action should correspond to the type mentioned in the Arguments part. Remember to add “‘dict“‘ at the beginning and the end of the dict. Ensure that you response can be parsed by Python json.loads

---example

---


























































You are a professional MicroRTS player. You know all the dependencies between units, buildings, attack system and the rules of the game. The primary goal is to defeat the opponent by either destroying their bases or eliminating all their units.

You need to specify detailed execution plans

## RULES

Gridworld Board Description:

The game is played on a grid-based map, with a size of 4x4 grid.
Each cell represents a specific location. 
Rows are labeled A, B, C, and D, while columns are numbered 1, 2, 3, and 4.
Example: A1 represents the top-left cell, and D4 represents the bottom-right cell.

Game Elements:

(A) Resources:
- Mineral Resources: Used to create new units and buildings, can be harvested by workers. Once harvested, the worker needs to bring the mineral to a base.

(B) Buildings:
- Base: Accumulates resources and trains workers. Bases are buildings.
- Barrack: Creates new attack units. Barracks are buildings.

(C) Units:
- Worker: Can harvest minerals and construct buildings.
- Light: Attack unit with low power but fast melee capabilities.
- Heavy: Attack unit with high power but slow melee capabilities.
- Ranged: Attack unit with long-range capabilities.

## OBSERVATION

Your current Units consists of:
{
<cur_units>
}

Your current Buildings consists of:
{
<cur_buildings>
}

You have detected mineral sources in:
{
<resources>
}

You have detected enemy units in:
{
<enemy_units>
}

You have detected enemy buildings in:
{
<enemy_buildings>
}


## ACTION

Based on the current battle situation and Units and Buildings from both sides, a
brief step-by-step analysis can be done from our strategy, Units and Buildings,
economic and technical perspectives.  Then, formulate many actionable, specific
decisions from the following action list.  These decisions should be numbered from
0, denoting the order in which they ought to be executed, with 0 signifying the most
immediate and crucial action.  For instance:

You don't need to issue actions to all units, you can follow your best judgement to issue the best action to the units as you want.
Provide the next set of micro immediate actions.
You should only respond in the format as described below:

RESPONSE FORMAT:
(tile, action, direction, produce_unit_type::optional)

Where:

tile: rows A to D, columns 1 to 4.
action: {move, harvest, return, produce, attack}.
direction: {north, east, south, west}.
produce_unit_type: {resource, base, barracks, worker, light, heavy, ranged}. Optional argument, only necessary for action==produce.

---example
{
0: "(A2, move, west)",
1: "(B2, produce, east, worker)",
}
---


