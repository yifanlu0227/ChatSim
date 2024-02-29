from shapely.geometry import Polygon
import numpy as np


def is_projection_overlap(proj1, proj2):
    """检查两个投影是否重叠"""
    return max(proj1[0], proj2[0]) <= min(proj1[1], proj2[1])


def is_rectangles_overlap(rect1, rect2):
    """检查两个矩形是否重叠"""
    for i in range(4):
        # 计算当前矩形的一条边
        edge = rect1[i] - rect1[(i + 1) % 4]
        # 计算法线（垂直方向）
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)

        # 在法线方向上投影两个矩形
        proj1 = project_polygon_onto_axis(Polygon(rect1), axis)
        proj2 = project_polygon_onto_axis(Polygon(rect2), axis)

        # 检查投影是否重叠
        if not is_projection_overlap(proj1, proj2):
            return False

    # 检查rect2的边
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
    """
    根据车辆中心点轨迹计算车辆四个角点的轨迹。
    输入:
    - trajectory: 车辆中心点的轨迹，大小为(T, 2)
    - car_length: 车辆长度
    - car_width: 车辆宽度

    输出:
    - corners_trajectory: 车辆四个角点的轨迹，大小为(T, 4, 2)
    """
    T = trajectory.shape[0]
    corners_trajectory = np.zeros((T, 4, 2))

    for i in range(1, T):
        # 计算朝向（即速度方向）
        direction = trajectory[i] - trajectory[i - 1]
        direction /= np.linalg.norm(direction)

        # 计算垂直于朝向的向量
        perpendicular = np.array([-direction[1], direction[0]])

        # 计算四个角点相对于中心点的位置
        front = 0.5 * car_length * direction
        back = -0.5 * car_length * direction
        left = 0.5 * car_width * perpendicular
        right = -0.5 * car_width * perpendicular

        # 计算四个角点的绝对位置
        corners_trajectory[i, 0] = trajectory[i] + front + left
        corners_trajectory[i, 1] = trajectory[i] + front + right
        corners_trajectory[i, 2] = trajectory[i] + back + right
        corners_trajectory[i, 3] = trajectory[i] + back + left

    # 第一个时间步使用第二个时间步的朝向
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
