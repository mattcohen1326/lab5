"""robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
import copy
from controller import Robot, Motor, DistanceSensor
import csci3302_lab5_supervisor
import numpy as np



state = "get_path"


#########################################################
# Starter code: You shouldn't have to modify any of this

# create the Robot instance.
csci3302_lab5_supervisor.init_supervisor()
robot = csci3302_lab5_supervisor.supervisor

# Map Variables
MAP_BOUNDS = [1.,1.] 
CELL_RESOLUTIONS = np.array([0.1, 0.1]) # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

def populate_map(m):
    obs_list = csci3302_lab5_supervisor.supervisor_get_obstacle_positions()
    obs_size = 0.06 # 6cm boxes
    for obs in obs_list:
        obs_coords_lower = obs - obs_size/2.
        obs_coords_upper = obs + obs_size/2.
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1
        obs_coords_lower = [obs[0] - obs_size/2, obs[1] + obs_size/2.]
        obs_coords_upper = [obs[0] + obs_size/2., obs[1] - obs_size/2.]
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1


# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3


EPUCK_MAX_WHEEL_SPEED = 0.12880519 # m/s
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.


# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

MAX_VEL_REDUCTION = 0.25


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    '''
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    '''
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    pose_theta = get_bounded_theta(pose_theta)

def get_bounded_theta(theta):
    '''
    Returns theta bounded in [-PI, PI]
    '''
    while theta > math.pi: theta -= 2.*math.pi
    while theta < -math.pi: theta += 2.*math.pi
    return theta

def get_wheel_speeds(target_pose):
    '''
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    '''
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = csci3302_lab5_supervisor.supervisor_get_robot_pose()


    bearing_error = math.atan2( (target_pose[1] - pose_y), (target_pose[0] - pose_x) ) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x,pose_y]))
    heading_error = target_pose[2] -  pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0
    
    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer
    
    if distance_error < 0.05 and abs(heading_error) < 0.05:    
        left_speed_pct = 0
        right_speed_pct = 0
        
    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()


    # print("Current pose: [%5f, %5f, %5f]\t\t Target pose: [%5f, %5f, %5f]\t\t %5f %5f %5f\t\t  %3f %3f" % (pose_x, pose_y, pose_theta, target_pose[0], target_pose[1], target_pose[2], bearing_error, distance_error, get_bounded_theta(heading_error), left_wheel_direction, right_wheel_direction))

        
    return phi_l_pct, phi_r_pct



def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None
    
    return np.array([(col+0.5)*CELL_RESOLUTIONS[1], (row+0.5)*CELL_RESOLUTIONS[0]])


def display_map(m):
    """
    @param m: The world map matrix to visualize
    """
    m2 = copy.copy(m)
    robot_pos = transform_world_coord_to_map_coord([pose_x,pose_y])
    m2[robot_pos] = 8
    map_str = ""
    for row in range(m.shape[0]-1,-1,-1):
        for col in range(m.shape[1]):
            if m2[row,col] == 0: map_str += '[ ]'
            elif m2[row,col] == 1: map_str += '[X]'
            elif m2[row,col] == 2: map_str += '[+]'
            elif m2[row,col] == 3: map_str += '[G]'
            elif m2[row,col] == 4: map_str += '[S]'
            elif m2[row,col] == 8: map_str += '[r]'
            else: map_str += '[E]'

        map_str += '\n'

    print(map_str)
    print(' ')

# ^ Starter code: You shouldn't have to modify any of this ^
############################################################




###################
# Part 1.1
###################
def get_travel_cost(source_vertex, dest_vertex):
    """
    @param source_vertex: world_map coordinates for the starting vertex
    @param dest_vertex: world_map coordinates for the destination vertex
    @return cost: Cost to travel from source to dest vertex.
    """
    global world_map
    cost = 0
    if source_vertex == dest_vertex:
        return cost
    if world_map[dest_vertex[0]][dest_vertex[1]] == 1:
        cost = 1
    cost += abs(source_vertex[0]-dest_vertex[0]) + abs(source_vertex[1]-dest_vertex[1])
    return cost


def extractMin(x):
    x.sort(key=lambda x:x[1],reverse=True)
    return x.pop()
    
def validVertex(v):
    global world_map
    if v[0] >= 0 and v[0] < world_map.shape[0] and v[1] >= 0 and v[1] < world_map.shape[1]:
        return True
    return False   

def getNeighbors(v):
    neighbors = []
    for i in range(0,2):
        for j in range(0,2):
            new_neighbor = (v[0]+i,v[1]+j)
            if validVertex(new_neighbor):
                neighbors.append(new_neighbor)
    return neighbors
def getNeighborsNew(v):
    neighbors = []
    up = (v[0]+1,v[1])
    down = (v[0]-1,v[1])
    left = (v[0],v[1]-1)
    right = (v[0],v[1]+1)
    dirs = [up,down,right,left]
    for move in dirs:
        if validVertex(move):
            neighbors.append(move)
    return neighbors
###################
# Part 1.2
###################
def dijkstra(source_vertex):
    """
    @param source_vertex: Starting vertex for the search algorithm.
    @return prev: Data structure that maps every vertex to the coordinates of the previous vertex (along the shortest path back to source)
    """
    print("hm")
    global world_map
    
    # TODO: Initialize these variables
    dist = np.zeros([world_map.shape[0],world_map.shape[1]])
    prev = {}
    rows = world_map.shape[0]
    cols = world_map.shape[1]
    q_cost = [(source_vertex,0)]
    dist[source_vertex[0],source_vertex[1]] = 0
    for i in range (0, rows):
        for j in range(0, cols):
            v = [i,j]
            if v != source_vertex:
                dist[i][j] = float('inf')
                prev[(i,j)] = None
            
    #print("ok")
    while len(q_cost) != 0:
        u_tuple = extractMin(q_cost)
        u = u_tuple[0]
        neighbors = getNeighborsNew(u)
        #print(len(q_cost))
        for v in neighbors:
            alt = dist[u[0]][u[1]] + get_travel_cost(u,v)
            if alt < dist[v[0]][v[1]]: 
                dist[v[0]][v[1]] = alt
                prev[(v[0],v[1])] = u
                count = 0
                #print("ok2")
                q_cost.append((v,alt))    
    
    
    return prev


###################
# Part 1.3
###################
def reconstruct_path(prev, goal_vertex):
    """
    @param prev: Data structure mapping each vertex to the next vertex along the path back to "source" (from Dijkstra)
    @param goal_vertex: Map coordinates of the goal_vertex
    @return path: List of vertices where path[0] = source_vertex_ij_coords and path[-1] = goal_vertex_ij_coords
    """
 
    path = [goal_vertex]
    i = goal_vertex[0]
    j = goal_vertex[1]
    while (i,j) in prev:
        path.append(prev[(i,j)])
        temp_i = i
        temp_j = j
        i = prev[(temp_i,temp_j)][0]
        j = prev[(temp_i,temp_j)][1] 
    # Hint: Start at the goal_vertex and work your way backwards using prev until there's no "prev" left to follow.
    #       Then, reverse the list and return it!
    path.reverse()
    return path


###################
# Part 1.4
###################
def visualize_path(path):
    """
    @param path: List of graph vertices along the robot's desired path    
    """
    global world_map
    count = 0
    print(path)
    print(len(path))
    for key in path:
        if count == 0:
            world_map[key[0]][key[1]] = 4
        elif count == len(path)-1:
            world_map[key[0]][key[1]] = 3
        else:
            world_map[key[0]][key[1]] = 2
        count = count + 1
    # TODO: Set a value for each vertex along path in the world_map for rendering: 2 = Path, 3 = Goal, 4 = Start
    
    return

def main():
    global robot, state, sub_state, map
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    last_odometry_update_time = None

    # Keep track of which direction each wheel is turning
    left_wheel_direction = WHEEL_STOPPED
    right_wheel_direction = WHEEL_STOPPED

    # Important IK Variable storing final desired pose
    target_pose = None # Populated by the supervisor, only when the target is moved.


    # Sensor burn-in period
    for i in range(10): robot.step(SIM_TIMESTEP)

    start_pose = csci3302_lab5_supervisor.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = start_pose

    #dijkstra([3,5])
    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:
        # Odometry update code -- do not modify
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()

        # Get target location -- do not modify
        if target_pose is None:
            target_pose = csci3302_lab5_supervisor.supervisor_get_target_pose()
            world_map[transform_world_coord_to_map_coord(target_pose[:2])] = 3 # Goal vertex!
            print("New IK Goal Received! Target: %s" % str(target_pose))
            print("Current pose: [%5f, %5f, %5f]\t\t Target pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta, target_pose[0], target_pose[1], target_pose[2]))
            populate_map(world_map)
            display_map(world_map)

        


        if state == 'get_path':
            ###################
            # Part 2.1a
            ###################       
            # Compute a path from start to target_pose
            prev = dijkstra([0,0])
            prev = reconstruct_path(prev, (8,1))
            visualize_path(prev)
            display_map(world_map)       
            pass
        elif state == 'get_waypoint':
            ###################
            # Part 2.1b
            ###################       
            # Get the next waypoint from the path
            pass
        elif state == 'move_to_waypoint':
            ###################
            # Part 2.1c
            ###################       

            # Hint: Use the IK control function to travel to the current waypoint
            # Syntax/Hint for using the IK Controller:
            # lspeed, rspeed = get_wheel_speeds(target_wp)
            # leftMotor.setVelocity(lspeed)
            # rightMotor.setVelocity(rspeed)    
            pass
        else:
            # Stop
            left_wheel_direction, right_wheel_direction = 0, 0
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)    
            pass
            
            
        #display_map(world_map)
        print(world_map.shape[0])
    
if __name__ == "__main__":
    main()







