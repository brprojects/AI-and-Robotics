
import numpy as np
import matplotlib.pyplot as plt
import random

# Define constants
K_att = 1 # Attractive force constant
K_rep = 50 # Repulsive force constant
D = 20 # Distance of influence of obstacles
goal = [175, 175] # Goal position
start = [25, 25] # Start position
no_obstacles = 20 # Number of obstacles

# Define world and obstacles
world = np.zeros((200, 200))
for i in range(no_obstacles):
    size = np.random.randint(low=10, high=50)
    x = np.random.randint(low=10, high=180)
    y = np.random.randint(low=10, high=180)
    shapes = ['square', 'rectangle', 'circle']
    random_shape = random.choice(shapes)
    if random_shape == 'square':
        if (x < 27 and y < 27) or (x+size > 173 and y+size > 173):
            pass
        else:
            world[x:x+size, y:y+size] = 1

    elif random_shape == 'rectangle':
        dim1 = random.uniform(0.5, 2)
        dim2 = random.uniform(0.5, 2)
        if (x < 27 and y < 27) or (x+size*dim1 > 173 and y+size*dim2 > 173):
            pass
        else:
            world[x:x+int(size*dim1), y:y+int(size*dim2)] = 1

    elif random_shape == 'circle':
        if (x-size/2 < 27 and y-size/2 < 27) or (x+size/2 > 173 and y+size/2 > 173):
            pass
        else:
            for i in range(200):
                for j in range(200):
                    # Compute the distance of each point from the center of the circle
                    dist = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                    if dist <= size/2:
                        # Set the values inside the circle to 1
                        world[i, j] = 1


# Define function to calculate attractive force
def attractive_force(current_pos):
    attractive_force = K_att * (np.array(goal) - np.array(current_pos))
    # print('att {0}'.format(attractive_force))
    return attractive_force

# Define function to calculate repulsive force
def repulsive_force(current_pos):
    rep_force = np.zeros(2)
    for i in range(200):
        for j in range(200):
            if world[i, j] == 1:
                # dist = (np.array(current_pos) - np.array([i, j]))
                dist_mag = np.sqrt(((current_pos[0]-j)**2)+((current_pos[1]-i)**2))
                if dist_mag < D:
                    # print('hi')
                    rep_force[0] += K_rep * ((1/dist_mag) - (1/D)) * (current_pos[0] - j)/(dist_mag**2)
                    rep_force[1] += K_rep * ((1/dist_mag) - (1/D)) * (current_pos[1] - i)/(dist_mag**2)

    # print(rep_force)
    return rep_force

# Define function to calculate total force
def total_force(current_pos):
    total_force = attractive_force(current_pos) + repulsive_force(current_pos)
    # print(total_force)
    return total_force

# Define function to update position
def update_pos(current_pos):
    tot_force = total_force(current_pos)
    direction = np.arctan2(tot_force[1], tot_force[0])
    new_position = current_pos + 2 * np.array([np.cos(direction), np.sin(direction)])
    print(new_position)
    return new_position

# Define function to calculate attractive force
def special_attractive_force(current_pos):
    attractive_force = K_att * (np.array([175, 25]) - np.array(current_pos))
    return attractive_force

# Define function to calculate total force
def special_total_force(current_pos):
    total_force = special_attractive_force(current_pos) + repulsive_force(current_pos)

    return total_force

# Define function to update position
def special_update_pos(current_pos):
    tot_force = special_total_force(current_pos)
    direction = np.arctan2(tot_force[1], tot_force[0])
    new_position = current_pos + 2 * np.array([np.cos(direction), np.sin(direction)])
    print('stuck')
    print(new_position)
    return new_position

# Define function to calculate attractive force
def special2_attractive_force(current_pos):
    attractive_force = K_att * (np.array([25, 175]) - np.array(current_pos))
    return attractive_force

# Define function to calculate total force
def special2_total_force(current_pos):
    total_force = special2_attractive_force(current_pos) + repulsive_force(current_pos)

    return total_force

# Define function to update position
def special2_update_pos(current_pos):
    tot_force = special2_total_force(current_pos)
    direction = np.arctan2(tot_force[1], tot_force[0])
    new_position = current_pos + 2 * np.array([np.cos(direction), np.sin(direction)])
    print('stuck')
    print(new_position)
    return new_position

# Define function to simulate movement
def simulate_movement():
    pos = start
    path = [pos]
    stuck_count = 1
    while np.sqrt(((pos[0]-goal[0])**2)+((pos[1]-goal[1])**2)) > 5:
        pos = update_pos(pos)
        path.append(pos)
        if len(path) > 4 and path[-1][0] - path[-3][0] < 0.1 and path[-1][1] - path[-3][1] < 0.1:
            if stuck_count < 5:
                for i in range(10 * stuck_count):
                    pos = special_update_pos(pos)
                    path.append(pos)
                stuck_count += 1
            elif stuck_count < 10:
                for i in range(10 * (stuck_count-4)):
                    pos = special2_update_pos(pos)
                    path.append(pos)
                stuck_count += 1
            elif stuck_count == 10:
                break
    return path

# Simulate movement and plot results
path = simulate_movement()
plt.imshow(world, cmap='gray')
plt.plot([p[0] for p in path], [p[1] for p in path], 'r')
plt.plot(start[0], start[1], 'go', markersize=10)
plt.plot(goal[0], goal[1], 'bo', markersize=10)

# Set the x and y limits
plt.xlim(0, 200)
plt.ylim(0, 200)

# plt.show()
plt.savefig('../images/potential_field2.png')
