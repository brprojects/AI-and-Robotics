import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.imshow(gridmap, cmap='gray',vmin=0, vmax=1)
    # plt.show()
    plt.savefig(('../images/occupancy_map.png'))

# initializes a grid map with specified size and resolution
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

# converts a world coordinate pose to a map coordinate based on the grid map and the specified map resolution
def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0]
    new_pose[1] = np.round(pose[1]/map_res) + origin[1]
    return new_pose.astype(int)

# creates a homogeneous transformation matrix from a 2D pose (x, y, theta)
def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr

# converts raw laser scan ranges to 2D points in the robot's coordinate frame
def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

# converts the raw measurements acquired by the robot (ranges_raw) into the correspoding cells of the gridmap
def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

# converts the raw poses of the robot (poses_raw) into the correspoding cells of the gridmap.
def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose

# returns all the cells along a straight line between two points in the gridmap
def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l

# converts probability values to log-odds values
def prob2logodds(p):
    return np.log(p/(1-p))

# converts log-odds values to probability values
def logodds2prob(l):
    return 1 / (1 + np.exp(-l))

# computes the inverse sensor model probability for a map cell given an endpoint of a laser scan ray
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    if cell[0] == endpoint[0] and cell[1] == endpoint[1]:
        return prob_free
    else:
        return prob_occ

# performs grid mapping with known poses using laser scan ranges and robot poses. 
# It updates the occupancy grid map based on the inverse sensor model.
def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    for i in range(len(poses_raw)):
        ranges = ranges2cells(ranges_raw[i], poses_raw[i], occ_gridmap, map_res)
        pose = poses2cells(poses_raw[i], occ_gridmap, map_res)
        for i in range(len(ranges[0])):
            points = bresenham(pose[0], pose[1], ranges[0][i], ranges[1][i])
            for point in points:
                if point[0] < 400 and point[1] < 400:
                    occ_gridmap[point[0]][point[1]] = logodds2prob(prob2logodds(inv_sensor_model(point, points[-1], prob_occ, prob_free)) + prob2logodds(occ_gridmap[point[0]][point[1]]))
    return occ_gridmap

map_size = 100
map_res = 0.25

prior = 0.50
prob_occ = 0.90
prob_free = 0.35

# load data
# range until occupied object at each degree
ranges_raw = np.loadtxt("ranges.data", delimiter=',', dtype=float)

# (x,y) coordinate and angle of viewing
poses_raw = np.loadtxt("poses.data", delimiter=',', dtype=float)

# initialize gridmap
occ_gridmap = init_gridmap(map_size, map_res) + prior

occ_gridmap = grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior)
# print(occ_gridmap)

plot_gridmap(occ_gridmap)
