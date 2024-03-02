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
- Worker: Can harvest minerals and construct buildings. Is able to move one tile at a timestep, can only harvest adjacent tiles, can only construct buildings in adjacent tiles.
- Light: Attack unit with low power but fast melee capabilities. Is able to move one tile at a timestep, can attack enemies in adjacent tiles.
- Heavy: Attack unit with high power but slow melee capabilities. Is able to move one tile at a timestep, can attack enemies in adjacent tiles.
- Ranged: Attack unit with long-range capabilities. Is able to move one tile at a timestep, can attack enemies that are within a 3 tile radius of the current position.

## OBSERVATION

Your current Units consists of:
{
(A3, worker, idle)
}

Your current Buildings consists of:
{
(B2, base)
}

You have detected mineral sources in:
{
(A1, resource)
}

You have detected enemy units in:
{
none
}

You have detected enemy buildings in:
{
(D4, enemy_base)
}


## ACTION

Based on the current observation of the game state, Units and Buildings from both sides, a brief step-by-step analysis can be done from our strategy.
You can only issue micro actions to the units, meaning that before a certain worker is able to harverst resources at A1, it needs first to be in an adjacent tile, for example A2 or B1. 
Also, you can only issue actions for the current timestep.
You don't need to issue actions to all units, but you can. 
Use your best judgement and strategy to select the actions. Provide the next set of micro immediate actions. 

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


