import openai

import ipdb
import os
import re
import ast
import matplotlib.pyplot as plt
from chatsim.foreground.motion_tools.tools import (
    transform_node_to_lane,
    generate_vertices,
    visualize,
    find_closest_centerline,
    rot_and_trans,
    filter_forward_lane,
    inverse_rot_and_trans,
    rot_and_trans_bbox,
    filter_right_lane,
    filter_left_lane,
    visualize_placement,
    hermite_spline_once,
    hermite_spline_twice,
)
from chatsim.foreground.motion_tools.placement_iterative import vehicle_placement
from chatsim.foreground.motion_tools.check_collision import (
    check_collision_and_revise_static
)
import numpy as np
import random
# from motion_tracking import motion_tracking


def vehicle_motion(
    map_data,
    all_current_vertices,
    current_trajectory=[],
    placement_result=[],
    v_init=[],
    high_level_action_direction=[],
    high_level_action_speed=[],
    dt=0.4,
    total_len=10,
    idx=1,
):

    if placement_result[0] is None:
        return (None, "no placement")

    if idx > 0:
        current_position = placement_result
        transformed_map_data = rot_and_trans(map_data, current_position)
        transformed_all_current_vertices = rot_and_trans_bbox(
            all_current_vertices, current_position
        )
    else:
        transformed_map_data = map_data.copy()
        transformed_all_current_vertices = all_current_vertices.copy()

    transformed_map_data = filter_forward_lane(transformed_map_data)

    if high_level_action_direction == "turn left":
        transformed_map_data_dest = filter_left_lane(transformed_map_data)
    elif high_level_action_direction == "turn right":
        transformed_map_data_dest = filter_right_lane(transformed_map_data)

    if (high_level_action_direction == "turn left"
        or high_level_action_direction == "turn right"):
        destination_anchor = transformed_map_data_dest["centerline"][::5]
        print(destination_anchor)
        sorted_destination = destination_anchor[
            random.randint(0, len(destination_anchor) - 1)
        ]
        sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        sorted_destination = sorted_destination[:2]
    elif high_level_action_direction == "straight":
        sorted_destination_init = np.array([v_init * dt * total_len, 0])
        _, sorted_destination = find_closest_centerline(
            transformed_map_data, sorted_destination_init
        )

        sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        sorted_destination = (sorted_destination[0:2] + sorted_destination[2:4]) / 2


    attempts = 0
    success = False

    start = np.array([0, 0])
    end = np.array([sorted_destination[0], sorted_destination[1]])
    Vs = np.array([v_init, 0])  # 可以理解为速度，这里是水平方向

    Ve = (
        v_init
        * sorted_destination_direction
        / np.linalg.norm(sorted_destination_direction)
    )
    Ve = np.abs(Ve)

    coordinates = hermite_spline_once(
        start,
        end,
        Vs,
        Ve,
        agent=idx,
        time=1,
        input_map=map_data,
        post_transform=(idx > 0, current_position if idx > 0 else None),
        obj=all_current_vertices,
    )

    current_midpoint = coordinates[-int(len(coordinates) / 2)]
    midpoint_check_flag, closest_centerline = find_closest_centerline(
        transformed_map_data, current_midpoint
    )
    midpoint = (closest_centerline[0:2] + closest_centerline[2:4]) / 2
    midpoint_direction = closest_centerline[2:4] - closest_centerline[0:2]
    Vm = v_init * (midpoint_direction) / np.linalg.norm(midpoint_direction)
    Vm = np.abs(Vm)

    coordinates = hermite_spline_twice(
        start,
        end,
        midpoint,
        Vs,
        Ve,
        Vm,
        agent=idx,
        time=2,
        input_map=map_data,
        post_transform=(idx > 0, current_position if idx > 0 else None),
        obj=all_current_vertices,
    )

    generated_trajectory = np.array(coordinates[:: int(len(coordinates) / total_len)])
    generated_trajectory = check_collision_and_revise_static(
        generated_trajectory, transformed_all_current_vertices
    )

    if idx > 0:
        generated_trajectory = inverse_rot_and_trans(
            generated_trajectory, current_position
        )
    else:
        generated_trajectory = np.array(coordinates[-total_len:])

    # file_name = "work_dirs/vis_gpt_traj/traj_" + str(idx) + "_final.png"

    # visualize(
    #     generated_trajectory,
    #     file_name=file_name,
    #     input_map=map_data,
    #     obj=all_current_vertices,
    # )

    return generated_trajectory

