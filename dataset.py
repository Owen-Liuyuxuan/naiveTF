#!/usr/bin/python3
"""Read cache json file and work as a pytorch dataset for planning Transformer.

Json file structure:
  str(frame_id) {
    frame: frame,
    objects: [
        {
            "id": object_id,
            "type": object_type as long,
            "transform": T [4, 4], - transform of the object in the world frame.
            "velocity": v [3], - velocity of the object in the world frame.
            "global_footprint": [4, 3] - footprint of the object in the world frame.
        },
        {},
    ],
    history_trajectories_transform_list: [N_h, 4, 4], past trajectory of the vehicle in the world frame. N=10,
    future_trajectories_transform_list: [N_f, 4, 4], future trajectory of the vehicle in the world frame. N=30,
    history_trajectories_speed_list: [N_h, 1], past speed of the vehicle in the world frame. N=10,
    future_trajectories_speed_list: [N_f, 1], future speed of the vehicle in the world frame. N=30,
    routes: [], list of lanelet2 id that is route of the vehicle in the neighborhood.
    nearby_lanelets_ids: [], list of lanelet2 id that is nearby the vehicle in the neighborhood.
  }

"""

import lanelet2
from autoware_lanelet2_extension_python.utility import (
    load_info_from_yaml,
    MapProjectorInfo,
)
from autoware_lanelet2_extension_python.projection import MGRSProjector
import autoware_lanelet2_extension_python.utility.utilities as utilities

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.spatial
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Union
from pathlib import Path
import glob


def automatic_find_projector_yaml(map_path):
    """in the same directory with the name projector_info.yaml"""
    # get the directory of the map file path
    map_dir = os.path.dirname(map_path)
    # get the name of the map file without the extension
    projector_info = os.path.join(map_dir, "map_projector_info.yaml")
    if os.path.exists(projector_info):
        return projector_info
    else:
        return None


LANELET_TYPE_MAPPING = {
    "road": 0,
    "private": 1,
    "highway": 2,
    "play_street": 3,
    "emergency_lane": 4,
    "bus_lane": 5,
    "bicycle_lane": 6,
    "exit": 7,
    "walkway": 8,
    "shared_walkway": 9,
    "crosswalk": 10,
    "stairs": 11,
    "road_shoulder": 12,
    "pedestrian_lane": 13,
    "bicycle_lane": 14,
    "none": 15,
}

LANELET_LOCATION_MAPPING = {"urban": 0, "nonurban": 1, "private": 2, "none": 3}

LANELET_TURN_DIRECTION_MAPPING = {"straight": 0, "left": 1, "right": 2}


def attribute_or(lanelet, key, default):
    if key in lanelet.attributes:
        return lanelet.attributes[key]
    return default


def get_lanelet2_projector(projector_info: MapProjectorInfo):
    """
    プロジェクタ情報に基づいて、適切なlanelet2のプロジェクタを返します。

    引数:
      projector_info: プロジェクタ情報を保持するオブジェクト

    戻り値:
      lanelet2のプロジェクタオブジェクト

    例外:
      ValueError: サポートされていないプロジェクタタイプが指定された場合
    """
    # LOCAL_CARTESIAN_UTM の場合
    if projector_info.projector_type == "LOCAL_CARTESIAN_UTM":
        position = lanelet2.GPSPoint(
            projector_info.map_origin.latitude,
            projector_info.map_origin.longitude,
            projector_info.map_origin.altitude,
        )
        origin = lanelet2.io.Origin(position)
        return lanelet2.projection.UtmProjector(origin)

    # MGRS の場合
    elif projector_info.projector_type == "MGRS":
        projector = MGRSProjector(lanelet2.io.Origin(0, 0))
        projector.setMGRSCode(projector_info.mgrs_grid)
        return projector

    # TRANSVERSE_MERCATOR の場合
    elif projector_info.projector_type == "TRANSVERSE_MERCATOR":
        position = lanelet2.core.GPSPoint(
            projector_info.map_origin.latitude,
            projector_info.map_origin.longitude,
            projector_info.map_origin.altitude,
        )
        origin = lanelet2.Origin(position)
        return TransverseMercatorProjector(origin)

    # LOCAL_CARTESIAN の場合
    elif projector_info.projector_type == "LOCAL_CARTESIAN":
        position = lanelet2.core.GPSPoint(
            projector_info.map_origin.latitude,
            projector_info.map_origin.longitude,
            projector_info.map_origin.altitude,
        )
        origin = lanelet2.ioOrigin(position)
        return lanelet2.projection.LocalCartesianProjector(origin)


def fit_rotated_bbox(points: np.ndarray) -> np.ndarray:
    """
    Fit a minimum-area rotated bounding box around a set of points.

    Args:
        points: numpy array of shape (N, 2) or (N, 3) containing coordinates

    Returns:
        corners: numpy array of shape (4, 3) containing the corners of the rotated bounding box
    """
    # Convert to numpy if tensor
    if isinstance(points, torch.Tensor):
        points = points.numpy()

    # Store Z coordinates if they exist
    has_z = points.shape[1] > 2
    if has_z:
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_val = z_min  # We'll use min z for the bbox corners
        points_2d = points[:, :2]
    else:
        z_val = 0
        points_2d = points

    # Remove duplicate 2D points
    points_2d = np.unique(points_2d, axis=0)

    # If we have less than 3 points, create a simple box
    if len(points_2d) < 3:
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        corners = np.array(
            [
                [min_coords[0], min_coords[1], z_val],
                [max_coords[0], min_coords[1], z_val],
                [max_coords[0], max_coords[1], z_val],
                [min_coords[0], max_coords[1], z_val],
            ]
        )
        return torch.tensor(corners, dtype=torch.float32)

    try:
        # Try normal convex hull first
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except scipy.spatial._qhull.QhullError:
        # If that fails, try with jitter
        jitter = np.random.normal(0, 1e-8, points_2d.shape)
        points_jittered = points_2d + jitter
        try:
            hull = ConvexHull(points_jittered)
            hull_points = points_2d[hull.vertices]
        except scipy.spatial._qhull.QhullError:
            # If still fails, fall back to axis-aligned bounding box
            min_coords = np.min(points_2d, axis=0)
            max_coords = np.max(points_2d, axis=0)
            corners = np.array(
                [
                    [min_coords[0], min_coords[1], z_val],
                    [max_coords[0], min_coords[1], z_val],
                    [max_coords[0], max_coords[1], z_val],
                    [min_coords[0], max_coords[1], z_val],
                ]
            )
            return torch.tensor(corners, dtype=torch.float32)

    # Get the edges of the convex hull
    edges = np.roll(hull_points, -1, axis=0) - hull_points

    # Normalize the edges
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    # Try all unique angles to find minimum area
    min_area = float("inf")
    best_corners = None

    for angle in angles:
        # Rotate points
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated_points = np.dot(points_2d, rotation)

        # Get axis-aligned bounding box
        min_x, min_y = np.min(rotated_points, axis=0)
        max_x, max_y = np.max(rotated_points, axis=0)

        # Calculate area
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area

            # Generate corners of the bounding box
            corners_2d = np.array(
                [
                    [min_x, min_y],  # bottom-left
                    [max_x, min_y],  # bottom-right
                    [max_x, max_y],  # top-right
                    [min_x, max_y],  # top-left
                ]
            )

            # Rotate corners back
            rotation_inv = np.array(
                [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
            )
            best_corners = np.dot(corners_2d, rotation_inv)

    # Add Z coordinate
    best_corners_3d = np.column_stack([best_corners, np.full(4, z_val)])

    return torch.tensor(best_corners_3d, dtype=torch.float32)


def identify_current_light_status(
    turn_direction: int, traffic_light_elements: list
) -> int:
    """
    Identify the current traffic light status based on turn direction and traffic light elements.

    Args:
        turn_direction: Integer representing the turn direction (0=straight, 1=left, 2=right)
        traffic_light_elements: List of dictionaries containing traffic light information
                               (color, shape, status, confidence)

    Returns:
        int: The color of the relevant traffic light (0=UNKNOWN, 1=RED, 2=AMBER, 3=GREEN, 4=WHITE)
    """
    # Filter out ineffective elements (color == 0)
    effective_elements = [
        element for element in traffic_light_elements if element["color"] != 0
    ]

    # If no effective elements, return UNKNOWN (0)
    if not effective_elements:
        return 0

    # If only one effective element, return its color
    if len(effective_elements) == 1:
        return effective_elements[0]["color"]

    # For multiple elements, find the one that matches the turn direction
    # Map turn direction to corresponding arrow shape
    direction_to_shape_map = {
        0: 4,  # straight -> UP_ARROW
        1: 2,  # left -> LEFT_ARROW
        2: 3,  # right -> RIGHT_ARROW
    }

    target_shape = direction_to_shape_map.get(turn_direction, 0)

    # First priority: Find elements with exactly matching direction
    matching_elements = [
        element for element in effective_elements if element["shape"] == target_shape
    ]
    if matching_elements:
        # If multiple matching elements, take the one with highest confidence
        return max(matching_elements, key=lambda x: x["confidence"])["color"]

    # Second priority: Find circle elements
    circle_elements = [
        element for element in effective_elements if element["shape"] == 1
    ]  # CIRCLE
    if circle_elements:
        # If multiple circle elements, take the one with highest confidence
        return max(circle_elements, key=lambda x: x["confidence"])["color"]

    # If no matching direction or circle, return the element with highest confidence
    return max(effective_elements, key=lambda x: x["confidence"])["color"]


def get_current_traffic_light(lanelet, current_traffic_light_status: dict):
    traffic_light_ids_connected_to_lanelet = lanelet.trafficLights()
    if len(traffic_light_ids_connected_to_lanelet) == 0:
        return 0  # unknown, empty

    main_traffic_light_id = str(traffic_light_ids_connected_to_lanelet[0].id)
    if main_traffic_light_id in current_traffic_light_status:
        current_traffic_group = current_traffic_light_status[main_traffic_light_id]
        turn_direction = LANELET_TURN_DIRECTION_MAPPING[
            attribute_or(lanelet, "turn_direction", "straight")
        ]
        light = identify_current_light_status(turn_direction, current_traffic_group)
        return light
    else:
        return 0 # the traffic light not detected


class CacheDataset(Dataset):
    """Read cache json file(s) and work as a pytorch dataset"""

    def __init__(
        self,
        cache_files: Union[str, List[str]],
        map_file: str,
        max_objects=80,
        max_map_element_num=80,
        max_route_element_num=10,
        lane_point_number=20,
    ):
        """
        Initialize dataset with single cache file or multiple cache files

        Args:
            cache_files: Either a single cache file path, a list of cache file paths,
                        or a directory containing cache files
            map_file: Path to the map file
            max_objects: Maximum number of objects to consider
            max_map_element_num: Maximum number of map elements
            max_route_element_num: Maximum number of route elements
            lane_point_number: Number of points per lane
        """
        self.cache = {}

        # Handle different input types for cache_files
        if isinstance(cache_files, str):
            path = Path(cache_files)
            if path.is_dir():
                # If directory, get all .json files
                cache_files = list(path.glob("*.json"))
            else:
                cache_files = [path]

        # Load all cache files
        for cache_file in cache_files:
            with open(cache_file, "r") as f:
                current_cache = json.load(f)
                # Add file identifier to frame_id to avoid collisions
                file_id = hash(str(cache_file)) % 10000  # Use hash of filename
                current_cache = {
                    f"{file_id}_{k}": v
                    for k, v in current_cache.items()
                    if k not in ["lanelet2_map", "vehicle_params"]
                }
                self.cache.update(current_cache)

        self.map_path = map_file
        self.projector_yaml = automatic_find_projector_yaml(map_file)
        if self.projector_yaml is None:
            self.projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        else:
            self.projector = get_lanelet2_projector(
                load_info_from_yaml(self.projector_yaml)
            )
        self.map_object = lanelet2.io.load(self.map_path, self.projector)

        self.keys = list(self.cache.keys())
        self.keys.sort(key=lambda x: int(x.split("_")[-1]))  # Sort by original frame id
        self.max_objects = max_objects
        self.max_map_element_num = max_map_element_num
        self.max_route_element_num = max_route_element_num
        self.lane_point_number = lane_point_number
        self._preprocess_map()

    def get_original_frame_id(self, index):
        """Get the original frame ID without the file identifier"""
        key = self.keys[index]
        return int(key.split("_")[-1])

    def _preprocess_map(self):
        """For each road lanelet in the map, resample the left and right boundary to a fix shape and generate the centerline"""
        self.resampled_lanelets = dict()
        for lanelet in self.map_object.laneletLayer:
            self.resampled_lanelets[int(lanelet.id)] = dict()
            left_length = lanelet2.geometry.length(lanelet.leftBound)
            right_length = lanelet2.geometry.length(lanelet.rightBound)
            length = max(left_length, right_length)
            if length < 0.001:
                continue
            resolution = (
                length / (self.lane_point_number - 1) + 0.0001
            )  # make sure the distance/resolution == self.lane_point_number
            right_bound = utilities.getRightBoundWithOffset(lanelet, 0.0, resolution)
            left_bound = utilities.getLeftBoundWithOffset(lanelet, 0.0, resolution)
            center_line = utilities.generateFineCenterline(lanelet, resolution)

            self.resampled_lanelets[int(lanelet.id)]["right_bound"] = right_bound
            self.resampled_lanelets[int(lanelet.id)]["left_bound"] = left_bound
            self.resampled_lanelets[int(lanelet.id)]["center_line"] = center_line

    def __getitem__(self, index):
        """Return the frame at the given index"""
        frame_id = self.keys[index]
        frame = self.cache[str(frame_id)]
        data = dict()
        data["frame_id"] = frame_id

        data["objects_types"] = torch.zeros([self.max_objects], dtype=torch.long)
        data["objects_mask"] = torch.zeros([self.max_objects], dtype=torch.bool)
        data["objects_transform"] = torch.zeros(
            [self.max_objects, 4, 4], dtype=torch.float32
        )
        data["objects_velocity"] = torch.zeros(
            [self.max_objects, 3], dtype=torch.float32
        )
        data["objects_footprint"] = torch.zeros(
            [self.max_objects, 4, 3], dtype=torch.float32
        )  # [x, y, z]

        actual_objects_num = len(frame["objects"])
        if actual_objects_num > self.max_objects:
            frame["objects"] = frame["objects"][: self.max_objects]
            actual_objects_num = self.max_objects

        if actual_objects_num > 0:
            data["objects_types"][:actual_objects_num] = torch.tensor(
                [object["type"] for object in frame["objects"]], dtype=torch.long
            )
            data["objects_mask"][:actual_objects_num] = torch.tensor(
                [True] * actual_objects_num, dtype=torch.bool
            )
            data["objects_transform"][:actual_objects_num] = torch.tensor(
                [object["transform"] for object in frame["objects"]],
                dtype=torch.float32,
            )
            data["objects_velocity"][:actual_objects_num] = torch.tensor(
                [object["velocity"] for object in frame["objects"]], dtype=torch.float32
            )

            # Process footprints
            # If the footprint is not a 4-point bbox, fit a rotated bbox to it
            processed_footprints = []
            for obj in frame["objects"][:actual_objects_num]:
                footprint = obj["global_footprint"]  # Shape could be (N, 2)
                if len(footprint) != 4:  # If not already a 4-point bbox
                    bbox = fit_rotated_bbox(np.array(footprint))
                    processed_footprints.append(bbox)
                else:
                    processed_footprints.append(
                        torch.tensor(footprint, dtype=torch.float32)
                    )

            data["objects_footprint"][:actual_objects_num] = torch.stack(
                processed_footprints
            )

        data["history_trajectories_transform"] = torch.tensor(
            frame["history_trajectories_transform_list"]
        )  # [N_h, 4, 4]
        data["future_trajectories_transform"] = torch.tensor(
            frame["future_trajectories_transform_list"]
        )  # [N_f, 4, 4]
        data["history_trajectories_speed"] = torch.tensor(
            frame["history_trajectories_speed_list"]
        )  # [N_h, 3]
        data["future_trajectories_speed"] = torch.tensor(
            frame["future_trajectories_speed_list"]
        )  # [N_f, 3]

        nearby_road_ids = [
            lanelet_idx
            for lanelet_idx in frame["nearby_lanelets_ids"]
            if int(lanelet_idx) in self.resampled_lanelets
        ]
        is_road_in_route = [
            bool(lanelet_idx in frame["routes"]) for lanelet_idx in nearby_road_ids
        ]
        if len(nearby_road_ids) > self.max_map_element_num:
            nearby_road_ids = nearby_road_ids[: self.max_map_element_num]
            is_road_in_route = is_road_in_route[: self.max_map_element_num]

        nearby_lanelets = [
            self.map_object.laneletLayer.get(lanelet_idx)
            for lanelet_idx in nearby_road_ids
        ]

        data["boundary_left_boundaries"] = torch.zeros(
            [self.max_map_element_num, self.lane_point_number, 2], dtype=torch.float32
        )
        data["boundary_right_boundaries"] = torch.zeros(
            [self.max_map_element_num, self.lane_point_number, 2], dtype=torch.float32
        )
        data["boundary_in_route"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.bool
        )
        data["boundary_mask"] = torch.zeros(
            [self.max_map_element_num, self.lane_point_number], dtype=torch.bool
        )
        data["lanelet_subtypes"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.long
        )
        data["lanelet_locations"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.long
        )
        data["lanelet_turn_directions"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.long
        )
        data["lanelet_speed_limit"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.float32
        )
        data["lanelet_traffic_lights"] = torch.zeros(
            [self.max_map_element_num], dtype=torch.long
        )

        data["boundary_mask"][0 : len(nearby_road_ids)] = True

        data["boundary_in_route"][0 : len(nearby_road_ids)] = torch.tensor(
            is_road_in_route, dtype=torch.bool
        )
        data["lanelet_subtypes"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                LANELET_TYPE_MAPPING[attribute_or(lanelet, "subtype", "road")]
                for lanelet in nearby_lanelets
            ],
            dtype=torch.long,
        )
        data["lanelet_locations"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                LANELET_LOCATION_MAPPING[attribute_or(lanelet, "location", "urban")]
                for lanelet in nearby_lanelets
            ],
            dtype=torch.long,
        )
        data["lanelet_turn_directions"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                LANELET_TURN_DIRECTION_MAPPING[
                    attribute_or(lanelet, "turn_direction", "straight")
                ]
                for lanelet in nearby_lanelets
            ],
            dtype=torch.long,
        )
        data["lanelet_speed_limit"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                float(attribute_or(lanelet, "speed_limit", "50"))
                / 100.0  # normalized to [0, 1]
                for lanelet in nearby_lanelets
            ],
            dtype=torch.float32,
        )
        data["lanelet_traffic_lights"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                get_current_traffic_light(
                    lanelet, frame["current_traffic_light_status"]
                )
                for lanelet in nearby_lanelets
            ],
            dtype=torch.long,
        )
        data["boundary_left_boundaries"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                [
                    (p.x, p.y)
                    for p in self.resampled_lanelets[int(lanelet_idx)]["left_bound"]
                ]
                for lanelet_idx in nearby_road_ids
            ]
        )
        data["boundary_right_boundaries"][0 : len(nearby_road_ids)] = torch.tensor(
            [
                [
                    (p.x, p.y)
                    for p in self.resampled_lanelets[int(lanelet_idx)]["right_bound"]
                ]
                for lanelet_idx in nearby_road_ids
            ]
        )

        ego_transform = data["history_trajectories_transform"][-1]  # [4, 4]
        ## normalize and transform the ego_trajectory, velocity, and neighbouring
        ego_transform = torch.inverse(ego_transform)
        data["future_trajectories_transform"] = torch.matmul(
            ego_transform, data["future_trajectories_transform"]
        )  # [N_f, 4, 4]
        ## ego speed is originally not in global frame, so we don't need to transform it
        data["history_trajectories_transform"] = torch.matmul(
            ego_transform,
            data["history_trajectories_transform"],
        )  # [N_h, 4, 4]
        data["objects_transform"] = torch.matmul(
            ego_transform, data["objects_transform"]
        )  # [N_o, 4, 4]

        # Transform velocities - only need rotation part of transformation
        rotation_matrix = ego_transform[:3, :3]  # [3, 3]
        data["objects_velocity"] = torch.matmul(
            rotation_matrix, data["objects_velocity"][..., None]
        )[
            ..., 0
        ]  # [N_o, 3]

        # Transform footprints from global to ego frame
        # First convert 2D points to homogeneous coordinates
        footprints_homogeneous = torch.ones(
            data["objects_footprint"].shape[0],
            data["objects_footprint"].shape[1],
            4,
            dtype=torch.float32,
        )  # [N_o, 4, 4]
        footprints_homogeneous[..., :3] = data[
            "objects_footprint"
        ]  # Set x,y coordinates
        footprints_homogeneous = footprints_homogeneous.transpose(-2, -1)  # Trans
        footprints_homogeneous = torch.matmul(
            ego_transform, footprints_homogeneous
        )  # [N_o, 4, 4]
        footprints_homogeneous = footprints_homogeneous.transpose(-2, -1)  # Trans
        data["objects_footprint"] = footprints_homogeneous[..., :2]  # [N_o, 4, 2]
        # print(data["objects_footprint"][data["objects_mask"]], ego_transform)

        # Transform map elements
        # Transform left boundaries
        left_boundaries_homogeneous = torch.ones(
            data["boundary_left_boundaries"].shape[0],
            data["boundary_left_boundaries"].shape[1],
            4,
            dtype=torch.float32,
        )
        left_boundaries_homogeneous[..., :2] = data["boundary_left_boundaries"]
        left_boundaries_homogeneous = left_boundaries_homogeneous.transpose(-2, -1)
        left_boundaries_homogeneous = torch.matmul(
            ego_transform, left_boundaries_homogeneous
        )
        data["boundary_left_boundaries"] = left_boundaries_homogeneous[
            :, :2, :
        ].transpose(-2, -1)

        # Transform right boundaries
        right_boundaries_homogeneous = torch.ones(
            data["boundary_right_boundaries"].shape[0],
            data["boundary_right_boundaries"].shape[1],
            4,
            dtype=torch.float32,
        )
        right_boundaries_homogeneous[..., :2] = data["boundary_right_boundaries"]
        right_boundaries_homogeneous = right_boundaries_homogeneous.transpose(-2, -1)
        right_boundaries_homogeneous = torch.matmul(
            ego_transform, right_boundaries_homogeneous
        )
        data["boundary_right_boundaries"] = right_boundaries_homogeneous[
            :, :2, :
        ].transpose(-2, -1)

        return data  # Return data instead of frame

    def __len__(self):
        """Return the number of frames in the cache"""
        return len(self.keys)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import numpy as np

    # Test loading and visualization
    cache_file = "/home/ukenryu/map_test/lanelet2_python_api_for_autoware/extract_bags/caches/cache.json"  # Replace with actual path
    map_file = "/home/ukenryu/autoware_map/odaiba_stable/lanelet2_map.osm"  # Replace with actual path

    dataset = CacheDataset(cache_file, map_file)

    # Get a sample frame
    sample_idx = 220
    data = dataset[sample_idx]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot ego vehicle position (should be at origin)
    ax.plot(0, 0, "r*", markersize=15, label="Ego Vehicle")

    # Plot other vehicles' footprints
    mask = data["objects_mask"]
    for i in range(mask.sum()):
        footprint = data["objects_footprint"][i]  # [4, 2]
        velocity = data["objects_velocity"][i]  # [3]
        object_type = data["objects_types"][i]

        # Plot footprint
        polygon = Polygon(footprint, fill=True, alpha=0.5)
        ax.add_patch(polygon)

        # Plot velocity vector
        center = footprint.mean(dim=0)
        ax.arrow(
            center[0],
            center[1],
            velocity[0],
            velocity[1],
            head_width=0.3,
            head_length=0.5,
            fc="red",
            ec="red",
        )

    # Plot map elements
    for i in range(data["boundary_mask"][..., 0].sum()):
        left = data["boundary_left_boundaries"][i]  # [lane_point_number, 2]
        right = data["boundary_right_boundaries"][i]  # [lane_point_number, 2]

        if data["boundary_in_route"][i]:
            ax.plot(left[:, 0], left[:, 1], "k-", linewidth=5, alpha=0.8)
            ax.plot(right[:, 0], right[:, 1], "k-", linewidth=5, alpha=0.8)
        else:
            ax.plot(left[:, 0], left[:, 1], "k-", linewidth=1, alpha=0.5)
            ax.plot(right[:, 0], right[:, 1], "k-", linewidth=1, alpha=0.5)

    # Plot ego trajectory
    history_positions = data["history_trajectories_transform"][:, :3, 3]  # [N_h, 3]
    # print(history_positions)
    future_positions = data["future_trajectories_transform"][:, :3, 3]  # [N_f, 3]

    ax.plot(
        history_positions[:, 0],
        history_positions[:, 1],
        "b-",
        linewidth=2,
        label="History",
    )
    ax.plot(
        future_positions[:, 0],
        future_positions[:, 1],
        "r--",
        linewidth=2,
        label="Future",
    )

    # Set equal aspect ratio and add labels
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f'Frame {data["frame_id"]}')
    # ax.legend()

    # Set reasonable view limits
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)

    # Add grid
    ax.grid(True)

    plt.show()
