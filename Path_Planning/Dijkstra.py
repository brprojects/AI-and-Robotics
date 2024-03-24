
import heapq
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbours = []
        self.g_score = float('inf')
        self.checked = 0

    # compares the g_score value
    def __lt__(self, other):
        return self.g_score < other.g_score

    def distance_to(self, other):
        # Euclidean distance
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


def astar(start_node, end_node):
    # counter = 0
    open_list = []
    heapq.heappush(open_list, start_node)
    came_from = {}
    start_node.g_score = 0
    start_node.checked = 1

    while len(open_list)>0:
        # counter += 1
        current_node = heapq.heappop(open_list)
        current_node.checked = 1

        if current_node == end_node:
            # print(counter)
            path = [current_node]
            while current_node in came_from:
                current_node = came_from[current_node]
                path.append(current_node)
            return path[::-1]

        for neighbour in current_node.neighbours:
            tentative_g_score = current_node.g_score + current_node.distance_to(neighbour)
            if tentative_g_score < neighbour.g_score:
                came_from[neighbour] = current_node
                neighbour.g_score = tentative_g_score
                if neighbour not in open_list:
                    heapq.heappush(open_list, neighbour)
    return None


def main():
    maze = np.load('.\maze.npy')
    height, width = maze.shape

    # Create a 2D grid of nodes
    grid = [[Node(x, y) for y in range(width)] for x in range(height)]

    # Populate the neighbours attribute for each node
    for x in range(height):
        for y in range(width):
            node = grid[x][y]

            # Add the neighbours
            if x > 0 and maze[x-1, y] != 1: #(-1,0)
                node.neighbours.append(grid[x - 1][y])
            if x < height - 1 and maze[x+1, y] != 1: #(1,0)
                node.neighbours.append(grid[x + 1][y])
            if y > 0 and maze[x, y-1] != 1: #(0,-1)
                node.neighbours.append(grid[x][y - 1])
            if y < width - 1 and maze[x, y+1] != 1: #(0,1)
                node.neighbours.append(grid[x][y + 1])
            if x > 0 and y > 0 and maze[x-1, y-1] != 1: #(-1,-1)
                node.neighbours.append(grid[x - 1][y - 1])
            if x < height - 1 and y > 0 and maze[x+1, y-1] != 1: #(1,-1)
                node.neighbours.append(grid[x + 1][y - 1])
            if x > 0 and y < width - 1 and maze[x-1, y+1] != 1: #(-1,1)
                node.neighbours.append(grid[x - 1][y + 1])
            if x < height - 1 and y < width - 1 and maze[x+1, y+1] != 1: #(1,1)
                node.neighbours.append(grid[x + 1][y + 1])

    # start_node = grid[46][8]
    # end_node = grid[0][45]

    start_node = grid[0][45]
    end_node = grid[46][8]

    path = astar(start_node, end_node)

    visits = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            if grid[x][y].checked == 1:
                visits[x, y] = 0.6
            elif maze[x][y] == 0:
                visits[x, y] = 1
    if path is not None:
        for i in path:
            visits[i.x, i.y] = 0.25


    plt.imshow(visits, cmap='gray')
    plt.show()
    # plt.savefig('../images/Dijkstra.png')

if __name__ == '__main__':
    main()
