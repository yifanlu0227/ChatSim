from shapely.geometry import Polygon
import numpy as np


def is_projection_overlap(proj1, proj2):

    return max(proj1[0], proj2[0]) <= min(proj1[1], proj2[1])


def is_rectangles_overlap(rect1, rect2):

    for i in range(4):

        edge = rect1[i] - rect1[(i + 1) % 4]

        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)

        proj1 = project_polygon_onto_axis(Polygon(rect1), axis)
        proj2 = project_polygon_onto_axis(Polygon(rect2), axis)


        if not is_projection_overlap(proj1, proj2):
            return False


    for i in range(4):
        edge = rect2[i] - rect2[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)

        proj1 = project_polygon_onto_axis(Polygon(rect1), axis)
        proj2 = project_polygon_onto_axis(Polygon(rect2), axis)

        if not is_projection_overlap(proj1, proj2):
            return False

    return True


def calculate_car_corners(trajectory, car_length=4.5, car_width=2):
    
    T = trajectory.shape[0]
    corners_trajectory = np.zeros((T, 4, 2))

    for i in range(1, T):
        direction = trajectory[i] - trajectory[i - 1]
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)
        else:
            direction = np.array([1, 0]) #just assume a random direction, it's not a precise calculation

        perpendicular = np.array([-direction[1], direction[0]])

        front = 0.5 * car_length * direction
        back = -0.5 * car_length * direction
        left = 0.5 * car_width * perpendicular
        right = -0.5 * car_width * perpendicular

        corners_trajectory[i, 0] = trajectory[i] + front + left
        corners_trajectory[i, 1] = trajectory[i] + front + right
        corners_trajectory[i, 2] = trajectory[i] + back + right
        corners_trajectory[i, 3] = trajectory[i] + back + left

    corners_trajectory[0] = corners_trajectory[1]

    return corners_trajectory


def project_polygon_onto_axis(polygon, axis):
    min_projection = float("inf")
    max_projection = float("-inf")
    for point in polygon.exterior.coords:
        projection = (point[0] * axis[0] + point[1] * axis[1]) / np.linalg.norm(axis)
        min_projection = min(min_projection, projection)
        max_projection = max(max_projection, projection)
    return min_projection, max_projection


def calculate_translation_vector(rect1, rect2, direction):
    axes = []
    for rect in [rect1, rect2]:
        for i in range(len(rect.exterior.coords) - 1):
            edge = np.subtract(rect.exterior.coords[i + 1], rect.exterior.coords[i])
            normal = [-edge[1], edge[0]]
            axes.append(normal / np.linalg.norm(normal))

    min_translation_vector = None
    min_translation_distance = float("inf")
    for axis in axes:
        min_proj_rect1, max_proj_rect1 = project_polygon_onto_axis(rect1, axis)
        min_proj_rect2, max_proj_rect2 = project_polygon_onto_axis(rect2, axis)

        if max_proj_rect1 < min_proj_rect2 or max_proj_rect2 < min_proj_rect1:
            # No overlap, no need to move
            return [0, 0]

        overlap = min(max_proj_rect1, max_proj_rect2) - max(
            min_proj_rect1, min_proj_rect2
        )
        translation_axis = np.multiply(axis, overlap)
        if np.dot(translation_axis, direction) < 0:
            translation_axis = np.multiply(translation_axis, -1)

        translation_distance = np.linalg.norm(translation_axis)
        if translation_distance < min_translation_distance:
            min_translation_distance = translation_distance
            min_translation_vector = translation_axis

    return min_translation_vector


def check_collision_and_revise_static(curr_trajectory, objects):
    # curr_trajectory (T,2)
    # objects (N,4,2)
    N, T = objects.shape[0], curr_trajectory.shape[0]
    if N == 0:
        return curr_trajectory
    car_length = 5
    car_width = 2.2
    safe_distance = 7
    obj_corners_trajectory = objects[:, None, :, :].repeat(T, axis=1)

    curr_corner_trajectory = calculate_car_corners(
        curr_trajectory, car_length, car_width
    )

    for j in range(N):
        for t in range(T):
            if is_rectangles_overlap(
                curr_corner_trajectory[t], obj_corners_trajectory[j, t]
            ):
                direction = curr_trajectory[t] - curr_trajectory[t - 1]
                direction /= np.linalg.norm(direction)

                perpendicular = np.array([-direction[1], direction[0]])
                delta = calculate_translation_vector(
                    Polygon(curr_corner_trajectory[t]),
                    Polygon(obj_corners_trajectory[j, t]),
                    perpendicular,
                )
                curr_trajectory[t] += delta

    return curr_trajectory



def check_collision_and_revise_dynamic(input_trajectory):
    # curr_trajectory (T,2)
    # objects (N,4,2)

    def judge_priority(traj1,traj2):
    # assume modify traj2
        T = traj1.shape[0]
        traj2_new = traj2[0:1].repeat(T,axis=0)

        for t in range(T):
            if is_rectangles_overlap(traj1[t],traj2_new[t]):
                return 1
        return 2

    def interpolate_uniformly(track, num_points):
        """
        Interpolates a given track to a specified number of points, distributing them uniformly.

        :param track: A numpy array of shape (n, d) where n is the number of points and d is the dimension.
        :param num_points: The number of points in the output interpolated track.
        :return: A numpy array of shape (num_points, d) representing the uniformly interpolated track.
        """
        # Calculate the cumulative distance along the track
        distances = np.cumsum(np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)  # Include the start point

        # Generate the desired number of equally spaced distances
        max_distance = distances[-1]
        uniform_distances = np.linspace(0, max_distance, num_points)

        # Interpolate for each dimension
        uniform_track = np.array([np.interp(uniform_distances, distances, track[:, dim]) for dim in range(track.shape[1])])

        return uniform_track.T
    
    def add_wait_timesteps(traj,t,wait_timesteps):
        traj_out = traj.copy()
        T = traj_out.shape[0]
        if t+wait_timesteps > T:
            traj_out = interpolate_uniformly(traj[:t],T)
        else:
            traj_out[:t+wait_timesteps] = interpolate_uniformly(traj[:t],t+wait_timesteps)
            traj_out[t+wait_timesteps:] = traj[t:-wait_timesteps]
        return traj_out
    

    valid_traj = []
    valid_record = []
    for item in input_trajectory:
        if item[0] is not None:
            valid_record.append(1)
            valid_traj.append(item)
        else:
            valid_record.append(0)

    curr_trajectory = np.array(valid_traj)
    car_length = 5
    car_width = 2.2
    safe_distance = 8
    N, T = curr_trajectory.shape[0], curr_trajectory.shape[1]
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(curr_trajectory[n], car_length, car_width)

    revised_trajectory = curr_trajectory.copy()

    for i in range(N):
        for j in range(i+1,N):
            for t in range(1,T):
                if is_rectangles_overlap(all_corners_trajectory[i,t],all_corners_trajectory[j,t]):
                    modify_idx = judge_priority(all_corners_trajectory[i],all_corners_trajectory[j])
                    collision_point = (curr_trajectory[i,t] + curr_trajectory[j,t]) / 2
                    if modify_idx == 1:
                        collision_speed = np.linalg.norm(curr_trajectory[j,t] - curr_trajectory[j,t-1])
                        wait_timesteps = int(np.ceil(safe_distance / (collision_speed + 0.0001))) 
                        curr_trajectory[i] = add_wait_timesteps(curr_trajectory[i],t,wait_timesteps)
                    if modify_idx == 2:
                        collision_speed = np.linalg.norm(curr_trajectory[i,t] - curr_trajectory[i,t-1])
                        wait_timesteps = int(np.ceil(safe_distance / (collision_speed + 0.0001))) 
                        curr_trajectory[j] = add_wait_timesteps(curr_trajectory[j],t,wait_timesteps)
                    break

    output = []
    num = 0
    for i in range(len(valid_record)):
        if valid_record[i] == 1:
            output.append(curr_trajectory[num])
            num += 1
        else:
            output.append(input_trajectory[i])
    return output
