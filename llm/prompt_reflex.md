"""You are a professional MicroRTS player. You know all the dependencies between units, buildings, attack system and the rules of the game. The primary goal is to defeat the opponent by either destroying their bases or eliminating all their units.

You need to specify detailed execution plans

## RULES

Gridworld Board Description:

The game is played on a grid-based map, with a size of 4x4 grid.
Each cell represents a specific location. 
Rows are labeled A, B, C, and D, while columns are numbered 0, 1, 2, and 3.
Example: A0 represents the top-left cell, and D3 represents the bottom-right cell.

Game Elements:

(A) Resources:
- Mineral Resources: Used to create new units and buildings, can be harvested by workers. Once harvested, the worker needs to bring the mineral to a base.

(B) Buildings:
- Base: Accumulates resources and trains workers. Bases are buildings.
- Barrack: Creates new attack units. Barracks are buildings.

(C) Units:
- Worker: Can harvest minerals and construct buildings. Is able to move one tile at a timestep, can only harvest adjacent tiles, can only construct buildings in adjacent tiles.
- Light: Attack unit with low power but fast melee capabilities. Is able to move one tile at a timestep, can attack enemies in adjacent tiles.
- Heavy: Attack unit with high power but slow melee capabilities. Is able to move one tile at a timestep, can attack enemies in adjacent tiles.
- Ranged: Attack unit with long-range capabilities. Is able to move one tile at a timestep, can attack enemies that are within a 3 tile radius of the current position.

## OBSERVATION


Obervation:
(position, unit_type, hp, resources, owner, action_type)
{
(A0, resource, hp:1, resources: 4, NEUTRAL, NOOP)
(B0, worker, hp:1, None, PLAYER, NOOP)
(B1, base, hp:4, None, PLAYER, NOOP)
(D3, base, hp:4, None, ENEMY, produce)
}

Set of valid actions:
(position, action_type, direction, production_type)
{
(A2, NOOP)
(A2, move, east)
(A2, move, south)
(A2, move, west)
(A2, produce, east, barracks)
(A2, produce, south, barracks)
(A2, produce, west, barracks)
}


## ACTION

Based on the current observation of the game state, Units and Buildings from both sides, a brief step-by-step analysis can be done from our strategy.
You can only issue micro actions to the units, meaning that before a certain worker is able to harverst resources at A1, it needs first to be in an adjacent tile, for example A2 or B1. 
Also, you can only issue actions for the current timestep.
Each unit can execute only one action, which means that in the response dictionary, you can only include one action per unit.
You don't need to issue actions to all units, but you can. 
You must respect the set of valid actions.
Use your best judgement and strategy to select the actions. Provide the next set of micro immediate actions. 

You should only respond in the format as described below:

RESPONSE FORMAT:
(tile, action, direction, produce_unit_type::optional)

Where:

tile: rows A to D, columns 0 to 3.
action: {move, harvest, return, produce, attack}.
direction: {north, east, south, west}.
produce_unit_type: {resource, base, barracks, worker, light, heavy, ranged}. Optional argument, only necessary for action==produce.

---example output:
{
0: "(A1, move, east)",
1: "(B1, produce, east, worker)",
}
---

"""
