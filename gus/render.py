
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = int(root.attrib['width'])
    height = int(root.attrib['height'])

    # Initialize an empty matrix
    matrix = np.zeros((height, width), dtype=int)
    #vector = np.zeros((height * width), dtype=int)

    # Process terrain
    terrain_data = root.find('terrain').text
    for i in range(height):
        for j in range(width):
            matrix[i, j] = int(terrain_data[i * width + j])
            #vector[i * width + j] = int(terrain_data[i * width + j])

    #print(vector)

    # Process units
    units = root.find('units')
    for unit in units.iter('rts.units.Unit'):
        unit_type = unit.attrib['type']
        player = int(unit.attrib['player'])

        x = int(unit.attrib['x'])
        y = int(unit.attrib['y'])

        if unit_type == 'Base':
            matrix[y, x] = player + 3  # Player bases represented by player + 2
        elif unit_type == 'Resource':
            matrix[y, x] = 2  # Resource tiles represented by -1
        elif unit_type == 'Worker':
            matrix[y, x] = player + 5  # Already built workers represented by player + 4
        elif unit_type == 'Barracks':
            matrix[y, x] = player + 7  # Barracks by player + 6
        elif unit_type == 'Light':
            matrix[y, x] = player + 9
        elif unit_type == 'Heavy':
            matrix[y, x] = player + 11
        elif unit_type == 'Ranged':
            matrix[y, x] = player + 13

    return matrix

def create_xml(matrix, xml_file):
    root = ET.Element("rts.PhysicalGameState")
    root.set("width", str(matrix.shape[1]))
    root.set("height", str(matrix.shape[0]))

    # Process terrain
    terrain_data = ""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                terrain_data += "1"
            else:
                terrain_data += "0"
    terrain = ET.SubElement(root, "terrain")
    terrain.text = terrain_data

    # Set Players
    players = ET.SubElement(root, "players")
    player = ET.SubElement(players, "rts.Player")
    player.set("ID", "0")
    player.set("resources", "0")
    player = ET.SubElement(players, "rts.Player")
    player.set("ID", "1")
    player.set("resources", "0")

    # Process units
    units = ET.SubElement(root, "units")
    id = 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            unit_type = matrix[i, j]
            if unit_type >= 2:
                unit = ET.SubElement(units, "rts.units.Unit")
                unit.set("type", get_unit_type(unit_type))
                unit.set("player", str(int((unit_type+1) % 2)))
                unit.set("x", str(j))
                unit.set("y", str(i))
                unit.set("resources", "0")
                unit.set("hitpoints", "4")
                unit.set("ID", str(id))
                id += 1
                #unit.text = '\n    '

    tree = ET.ElementTree(root)
    #tree.write(xml_file)
    ET.indent(tree, '  ')
    tree.write(xml_file, encoding="utf-8", short_empty_elements=False)

def get_unit_type(unit_type):
    if unit_type == 2:
        return "Resource"
    elif unit_type == 3 or unit_type == 4:
        return "Base"
    elif unit_type == 5 or unit_type == 6:
        return "Worker"
    elif unit_type == 7 or unit_type == 8:
        return "Barracks"
    elif unit_type == 9 or unit_type == 10:
        return "Light"
    elif unit_type == 11 or unit_type == 12:
        return "Heavy"
    elif unit_type == 13 or unit_type == 14:
        return "Ranged"

def plot_map(matrix):
    #cmap = plt.cm.Dark2
    #norm = plt.Normalize(matrix.min(), matrix.max())

    fig, ax = plt.subplots()
    ax.imshow(matrix)#, cmap=cmap)#), norm=norm)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:  # Empty tiles represented by 0
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
            elif matrix[i, j] == 1:  # Wall tiles represented by 1
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
            elif matrix[i, j] == 2:  # Resource tiles represented by -1
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='darkgreen')
                ax.add_patch(rect)
            elif matrix[i, j] == 3:  # Player 1 base represented by 2
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='blue', facecolor='lightblue')
                ax.add_patch(rect)
            elif matrix[i, j] == 4:  # Player 2 base represented by 3
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='red', facecolor=(1.0, 0.8, 0.8, 1.0))
                ax.add_patch(rect)
            elif matrix[i, j] == 5:  # Player 1 worker represented by 4
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.2, fill=True, edgecolor='blue', facecolor='grey')
                ax.add_patch(circle)
            elif matrix[i, j] == 6:  # Player 2 worker represented by 5
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.2, fill=True, edgecolor='red', facecolor='grey')
                ax.add_patch(circle)
            elif matrix[i, j] == 7:  # Player 1 barracks represented by 6
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='blue', facecolor='grey')
                ax.add_patch(rect)
            elif matrix[i, j] == 8:  # Player 2 barracks represented by 7
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='red', facecolor='grey')
                ax.add_patch(rect)
            elif matrix[i, j] == 9:  # Player 1 light represented by 8
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.3, fill=True, edgecolor='blue', facecolor='orange')
                ax.add_patch(circle)
            elif matrix[i, j] == 10:  # Player 2 light represented by 9
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.3, fill=True, edgecolor='red', facecolor='orange')
                ax.add_patch(circle)
            elif matrix[i, j] == 11:  # Player 1 heavy represented by 10
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.5, fill=True, edgecolor='blue', facecolor='yellow')
                ax.add_patch(circle)
            elif matrix[i, j] == 12:  # Player 2 heavy represented by 11
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.5, fill=True, edgecolor='red', facecolor='yellow')
                ax.add_patch(circle)
            elif matrix[i, j] == 13:  # Player 1 ranged represented by 12
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.4, fill=True, edgecolor='blue', facecolor='lightblue')
                ax.add_patch(circle)
            elif matrix[i, j] == 14:  # Player 2 ranged represented by 13
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor='lightgrey')
                ax.add_patch(rect)
                circle = plt.Circle((j, i), 0.4, fill=True, edgecolor='red', facecolor='lightblue')
                ax.add_patch(circle)

    ax.set_aspect('equal', 'box')
    plt.xticks(range(matrix.shape[1]))
    plt.yticks(range(matrix.shape[0]))
    plt.show()

def plot_action_mask(envs, map_size):
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_mask_tiles = action_mask.reshape(map_size, map_size, -1)
    plt.figure(figsize=(5, 5))
    plt.imshow(action_mask_tiles.sum(-1))

    plt.gca().set_xticks(np.arange(-.5, map_size, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, map_size, 1), minor=True)
    plt.grid(which="minor", axis="both", linestyle="-", color="k", linewidth=1)

    plt.xticks(np.arange(0, map_size, 1))
    plt.yticks(np.arange(0, map_size, 1))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Action Mask")
    plt.show()

def print_action_space(action_mask_tile):
    # (action_mask[0:6]),  # action type: {NOOP, move, harvest, return, produce, attack}
    # (action_mask[6:10]),  # move parameter: {north, east, south, west}
    # (action_mask[10:14]),  # harvest parameter: {north, east, south, west}
    # (action_mask[14:18]),  # return parameter: {north, east, south, west}
    # (action_mask[18:22]),  # produce_direction parameter: {north, east, south, west}
    # (action_mask[22:29]),  # produce_unit_type parameter: {resource, base, barracks, worker, light, heavy, ranged}
    # (action_mask[29 : sum(envs.action_space.nvec[1:])]),  # attack_target parameter: relative position of the unit that will be attacked

    # visualization of the action space for a given tile, translating where 1 means that the corresponding action is available
    
    action_types = ["NOOP", "move", "harvest", "return", "produce", "attack"]

    print("Action Type:")
    mask = action_mask_tile[0:6]
    print_masked(mask, mask)
    print_masked(mask, action_types)
    print()

    location_parameters = ["north", "east", "south", "west"]

    print("Move Parameter:")
    mask = action_mask_tile[6:10]
    print_masked(mask, mask)
    print_masked(mask, location_parameters)
    print()

    print("Harvest Parameter:")
    mask = action_mask_tile[10:14]
    print_masked(mask, mask)
    print_masked(mask, location_parameters)
    print()

    print("Return Parameter:")
    mask = action_mask_tile[14:18]
    print_masked(mask, mask)
    print_masked(mask, location_parameters)
    print()

    print("Produce Direction Parameter:")
    mask = action_mask_tile[18:22]
    print_masked(mask, mask)
    print_masked(mask, location_parameters)
    print()

    unit_types = ["resource", "base", "barracks", "worker", "light", "heavy", "ranged"]

    print("Produce Unit Type Parameter:")
    mask = action_mask_tile[22:29]
    print_masked(mask, mask)
    print_masked(mask, unit_types)
    print()

    print("Attack Target Parameter:")
    mask = action_mask_tile[29:sum(envs.action_space.nvec[1:])]
    print(mask.reshape(7, 7))
    print()

def print_masked(mask, params):
    for i, param in enumerate(params):
        if mask[i] == 1:
            print(f"\033[1m{param: ^7}\033[0m", end=" ")
        else:
            print(f'{param: ^7}', end=" ")
    print()



ac_move_east = np.array([
    1,  # action type: {NOOP, move, harvest, return, produce, attack}
    1,  # move parameter: {north, east, south, west}
    0,  # harvest parameter: {north, east, south, west}
    0,  # return parameter: {north, east, south, west}
    0,  # produce_direction parameter: {north, east, south, west}
    0,  # produce_unit_type parameter: {resource, base, barracks, worker, light, heavy, ranged}
    0  # attack_target parameter: relative position of the unit that will be attacked
])
ac_produce_worker_south = np.array([4, 0, 0, 0, 2, 3, 0])
ac_produce_worker_west = np.array([4, 0, 0, 0, 3, 3, 0])