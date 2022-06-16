__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import yaml
import random
import itertools
import functools

from pathlib import Path

from lib.OpenPCDet.pcdet.utils import calibration_kitti

from typing import Dict, List, Tuple
from tqdm.contrib.concurrent import process_map
from scipy.constants import speed_of_light as c     # in m/s

import numpy as np
import multiprocessing as mp
import tools.snowfall.geometry as g

from tools.wet_ground.planes import calculate_plane
from tools.wet_ground.augmentation import estimate_laser_parameters

PI = np.pi
DEBUG = False
EPSILON = np.finfo(float).eps



def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.parent.parent.absolute() / \
                 'lib' / 'OpenPCDet' / 'data' / 'dense' / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


def process_single_channel(root_path: str, particle_file_prefix: str, orig_pc: np.ndarray, beam_divergence: float,
                           order: List[int], channel_infos: List, channel: int) -> Tuple:
    """
    :param root_path:               Needed for training on GPU cluster.
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param orig_pc:                 N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param beam_divergence:         Equivalent to the total beam opening angle (in degree).
    :param order:                   Order of the particle disks.
    :param channel_infos            List of Dicts containing sensor calibration info.

    :param channel:                 Number of the LiDAR channel [0, 63].

    :return:                        Tuple of
                                    - intensity_diff_sum,
                                    - idx,
                                    - the augmented points of the current LiDAR channel.
    """

    intensity_diff_sum = 0

    index = order[channel]

    min_intensity = channel_infos[channel].get('min_intensity', 0)  # not all channels contain this info

    focal_distance = channel_infos[channel]['focal_distance'] * 100
    focal_slope = channel_infos[channel]['focal_slope']
    focal_offset = (1 - focal_distance / 13100) ** 2                # from velodyne manual

    particle_file = f'{particle_file_prefix}_{index + 1}.npy'

    channel_mask = orig_pc[:, 4] == channel
    idx = np.where(channel_mask == True)[0]

    pc = orig_pc[channel_mask]

    N = pc.shape[0]

    x, y, z, intensity, label = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], pc[:, 4]

    distance = np.linalg.norm([x, y, z], axis=0)

    center_angles = np.arctan2(y, x)  # in range [-PI, PI]
    center_angles[center_angles < 0] = center_angles[center_angles < 0] + 2 * PI  # in range [0, 2*PI]

    beam_angles = -np.ones((N, 2))

    beam_angles[:, 0] = center_angles - np.radians(beam_divergence / 2)  # could lead to negative values
    beam_angles[:, 1] = center_angles + np.radians(beam_divergence / 2)  # could lead to values above 2*PI

    # put beam_angles back in range [0, 2*PI]
    beam_angles[beam_angles < 0] = beam_angles[beam_angles < 0] + 2 * PI
    beam_angles[beam_angles > 2 * PI] = beam_angles[beam_angles > 2 * PI] - 2 * PI

    occlusion_list = get_occlusions(beam_angles=beam_angles, ranges_orig=distance, beam_divergence=beam_divergence,
                                    root_path=root_path, particle_file=particle_file)

    lidar_range = 120                       # in meter
    intervals_per_meter = 10                # => 10cm discretization
    beta_0 = 1 * 10 ** -6 / PI
    tau_h = 1e-8                            #  value 10ns taken from HDL64-S1 specsheet

    M = lidar_range * intervals_per_meter

    M_extended = int(np.ceil(M + c * tau_h * intervals_per_meter))
    lidar_range_extended = lidar_range + c * tau_h

    R = np.round(np.linspace(0, lidar_range_extended, M_extended), len(str(intervals_per_meter)))

    for j, beam_dict in enumerate(occlusion_list):

        d_orig = distance[j]
        i_orig = intensity[j]

        if channel in [53, 55, 56, 58]:
            max_intensity = 230
        else:
            max_intensity = 255

        i_adjusted = i_orig - 255 * focal_slope * np.abs(focal_offset - (1 - d_orig/120)**2)
        i_adjusted = np.clip(i_adjusted, 0, max_intensity)      # to make sure we don't get negative values

        CA_P0 = i_adjusted * d_orig ** 2 / beta_0

        if len(beam_dict.keys()) > 1:                           # otherwise there is no snowflake in the current beam

            i = np.zeros(M_extended)

            for key, tuple_value in beam_dict.items():

                if key != -1:                                   # if snowflake
                    i_orig = 0.9 * max_intensity                # set i to 90% reflectivity
                    CA_P0 = i_orig / beta_0                     # and do NOT normalize with original range

                r_j, ratio = tuple_value

                start_index = int(np.ceil(r_j * intervals_per_meter))
                end_index = int(np.floor((r_j + c * tau_h) * intervals_per_meter) + 1)

                for k in range(start_index, end_index):
                    i[k] += received_power(CA_P0, beta_0, ratio, R[k], r_j, tau_h)

            max_index = np.argmax(i)
            i_max = i[max_index]
            d_max = (max_index / intervals_per_meter) - (c * tau_h / 2)

            i_max += max_intensity * focal_slope * np.abs(focal_offset - (1 - d_max/120)**2)
            i_max = np.clip(i_max, min_intensity, max_intensity)

            if abs(d_max - d_orig) < 2 * (1 / intervals_per_meter):  # only alter intensity

                pc[j, 4] = 1

                new_i = int(i_max)

                if new_i > (i_orig + 1) and DEBUG:
                    print(f'\nnew intensity ({new_i}) in channel {channel} bigger than before ({i_orig}) '
                          f'=> clipping to {i_orig}')
                    new_i = np.clip(new_i, min_intensity, i_orig)
                    pc[j, 4] = 0

                intensity_diff_sum += i_orig - new_i

            else:  # replace point of hard target with snowflake

                pc[j, 4] = 2

                d_scaling_factor = d_max / d_orig

                pc[j, 0] = pc[j, 0] * d_scaling_factor
                pc[j, 1] = pc[j, 1] * d_scaling_factor
                pc[j, 2] = pc[j, 2] * d_scaling_factor

                new_i = int(i_max)

            assert new_i >= 0, f'new intensity is negative ({new_i})'

            clipped_i = np.clip(new_i, min_intensity, max_intensity)

            pc[j, 3] = clipped_i

        else:

            pc[j, 4] = 0

    return intensity_diff_sum, idx, pc


def binary_angle_search(angles: List[float], low: int, high: int, angle: float) -> int:
    """
    Adapted from https://www.geeksforgeeks.org/python-program-for-binary-search

    :param angles:                  List of individual endpoint angles.
    :param low:                     Start index.
    :param high:                    End index.
    :param angle:                   Query angle.

    :return:                        Index of angle if present in list of angles, else -1
    """

    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If angle is present at the middle itself
        if angles[mid] == angle:
            return mid

        # If angle is smaller than mid, then it can only be present in left sublist
        elif angles[mid] > angle:
            return binary_angle_search(angles, low, mid - 1, angle)

        # Else the angle can only be present in right sublist
        else:
            return binary_angle_search(angles, mid + 1, high, angle)

    else:
        # Angle is not present in the list
        return -1


def compute_occlusion_dict(beam_angles: Tuple[float, float], intervals: np.ndarray,
                           current_range: float, beam_divergence: float) -> Dict:
    """
    :param beam_angles:         Tuple of angles (left, right).
    :param intervals:           N-by-3 array of particle tangent angles and particle distance from origin.
    :param current_range:       Range to the original hard target.
    :param beam_divergence:     Equivalent to the total beam opening angle (in degree).

    :return:
    occlusion_dict:             Dict containing a tuple of the distance and the occluded angle by respective particle.
                                e.g.
                                0: (distance to particle, occlusion ratio [occluded angle / total angle])
                                1: (distance to particle, occlusion ratio [occluded angle / total angle])
                                3: (distance to particle, occlusion ratio [occluded angle / total angle])
                                7: (distance to particle, occlusion ratio [occluded angle / total angle])
                                ...
                                -1: (distance to original target, unocclusion ratio [unoccluded angle / total angle])

                                all (un)occlusion ratios always sum up to the value 1
    """

    try:
        N = intervals.shape[0]
    except IndexError:
        N = 1

    right_angle, left_angle = beam_angles

    # Make everything properly sorted in the corner case of phase discontinuity.
    if right_angle > left_angle:
        right_angle = right_angle - 2*PI
        right_left_order_violated = intervals[:, 0] > intervals[:, 1]
        intervals[right_left_order_violated, 0] = intervals[right_left_order_violated, 0] - 2*PI

    endpoints = sorted(set([right_angle] + list(itertools.chain(*intervals[:, :2])) + [left_angle]))
    diffs = np.diff(endpoints)
    n_intervals = diffs.shape[0]

    assignment = -np.ones(n_intervals)

    occlusion_dict = {}

    for j in range(N):

        a1, a2, distance = intervals[j]

        i1 = binary_angle_search(endpoints, 0, len(endpoints), a1)
        i2 = binary_angle_search(endpoints, 0, len(endpoints), a2)

        assignment_made = False

        for k in range(i1, i2):

            if assignment[k] == -1:
                assignment[k] = j
                assignment_made = True

        if assignment_made:
            ratio = diffs[assignment == j].sum() / np.radians(beam_divergence)
            occlusion_dict[j] = (distance, np.clip(ratio, 0, 1))

    ratio = diffs[assignment == -1].sum() / np.radians(beam_divergence)
    occlusion_dict[-1] = (current_range, np.clip(ratio, 0, 1))

    return occlusion_dict


def get_occlusions(beam_angles: np.ndarray, ranges_orig: np.ndarray, root_path: str, particle_file: str,
                   beam_divergence: float) -> List:
    """
    :param beam_angles:         M-by-2 array of beam endpoint angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :param ranges_orig:         M-by-1 array of original ranges corresponding to beams (in m).
    :param root_path:           Needed for training on GPU cluster.

    :param particle_file:       Path to N-by-3 array of all sampled particles as disks,
                                where each row contains abscissa and ordinate of the disk center and disk radius (in m).
    :param beam_divergence:     Equivalent to the opening angle of an individual LiDAR beam (in degree).

    :return:
    occlusion_list:             List of M Dicts.
                                Each Dict contains a Tuple of
                                If key == -1:
                                - distance to the original hard target
                                - angle that is not occluded by any particle
                                Else:
                                - the distance to an occluding particle
                                - the occluded angle by this particle

    """

    M = np.shape(beam_angles)[0]

    if root_path:
        path = Path(root_path) / 'training' / 'snowflakes' / 'npy' / particle_file
    else:
        path = Path(__file__).parent.parent.parent.absolute() / 'npy' / particle_file

    all_particles = np.load(str(path))
    x, y, _ = all_particles[:, 0], all_particles[:, 1], all_particles[:, 2]

    all_particle_ranges = np.linalg.norm([x, y], axis=0)                                                        # (N,)
    all_beam_limits_a, all_beam_limits_b = g.angles_to_lines(beam_angles)                                       # (M, 2)

    occlusion_list = []

    # Main loop over beams.
    for i in range(M):

        current_range = ranges_orig[i]                                                                          # (K,)

        right_angle = beam_angles[i, 0]
        left_angle = beam_angles[i, 1]

        in_range = np.where(all_particle_ranges < current_range)

        particles = all_particles[in_range]                                                                     # (K, 3)

        x, y, particle_radii = particles[:, 0], particles[:, 1], particles[:, 2]

        particle_angles = np.arctan2(y, x)                                                                      # (K,)
        particle_angles[particle_angles < 0] = particle_angles[particle_angles < 0] + 2 * PI

        tangents_a, tangents_b = g.tangents_from_origin(particles)                                              # (K, 2)

        ################################################################################################################
        # Determine whether centers of the particles lie inside the current beam,
        # which is first sufficient condition for intersection.
        standard_case = np.logical_and(right_angle <= particle_angles, particle_angles <= left_angle)
        seldom_case = np.logical_and.reduce((right_angle - 2 * PI <= particle_angles, particle_angles <= left_angle,
                                             np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))
        seldom_case_2 = np.logical_and.reduce((right_angle <= particle_angles, particle_angles <= left_angle + 2 * PI,
                                               np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))

        center_in_beam = np.logical_or.reduce((standard_case, seldom_case, seldom_case_2))  # (K,)
        ################################################################################################################

        ################################################################################################################
        # Determine whether distances from particle centers to beam rays are smaller than the radii of the particles,
        # which is second sufficient condition for intersection.
        beam_limits_a = all_beam_limits_a[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_b = all_beam_limits_b[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_c = np.zeros((2, 1))  # origin                                                              # (2, 1)

        # Get particle distances to right and left beam limit.
        distances = g.distances_of_points_to_lines(particles[:, :2],
                                                   beam_limits_a, beam_limits_b, beam_limits_c)                 # (K, 2)

        radii_intersecting = distances < np.column_stack((particle_radii, particle_radii))                      # (K, 2)

        intersect_right_ray = g.do_angles_intersect_particles(right_angle, particles[:, 0:2]).T                 # (K, 1)
        intersect_left_ray = g.do_angles_intersect_particles(left_angle, particles[:, 0:2]).T                   # (K, 1)

        right_beam_limit_hit = np.logical_and(radii_intersecting[:, 0], intersect_right_ray[:, 0])
        left_beam_limit_hit = np.logical_and(radii_intersecting[:, 1], intersect_left_ray[:, 0])

        ################################################################################################################
        # Determine whether particles intersect the current beam by taking the disjunction of the above conditions.
        particles_intersect_beam = np.logical_or.reduce((center_in_beam,
                                                         right_beam_limit_hit, left_beam_limit_hit))            # (K,)

        ################################################################################################################

        intersecting_beam = np.where(particles_intersect_beam)

        particles = particles[intersecting_beam]  # (L, 3)
        particle_angles = particle_angles[intersecting_beam]
        tangents_a = tangents_a[intersecting_beam]
        tangents_b = tangents_b[intersecting_beam]
        tangents = (tangents_a, tangents_b)
        right_beam_limit_hit = right_beam_limit_hit[intersecting_beam]
        left_beam_limit_hit = left_beam_limit_hit[intersecting_beam]

        # Get the interval angles from the tangents.
        tangent_angles = g.tangent_lines_to_tangent_angles(tangents, particle_angles)                           # (L, 2)

        # Correct tangent angles that do exceed beam limits.
        interval_angles = g.tangent_angles_to_interval_angles(tangent_angles, right_angle, left_angle,
                                                              right_beam_limit_hit, left_beam_limit_hit)        # (L, 2)

        ################################################################################################################
        # Sort interval angles by increasing distance from origin.
        distances_to_origin = np.linalg.norm(particles[:, :2], axis=1)                                          # (L,)

        intervals = np.column_stack((interval_angles, distances_to_origin))                                     # (L, 3)
        ind = np.argsort(intervals[:, -1])
        intervals = intervals[ind]                                                                              # (L, 3)

        occlusion_list.append(compute_occlusion_dict((right_angle, left_angle),
                                                     intervals,
                                                     current_range,
                                                     beam_divergence))

    return occlusion_list


def augment(pc: np.ndarray, particle_file_prefix: str, beam_divergence: float, shuffle: bool = True,
            show_progressbar: bool=False, only_camera_fov: bool=True, noise_floor: float=0.7,
            root_path: str=None) -> Tuple:
    """
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param show_progressbar:        Flag if tqdm should display a progessbar.
    :param only_camera_fov:         Flag if the camera field of view (FOV) filter should be applied.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        Tuple of
                                    - Tuple of the following statistics
                                        - num_attenuated,
                                        - avg_intensity_diff
                                    - N-by-4 array of the augmented pointcloud.
    """

    pc = pc[pc[:, 4].argsort()]             # sort pointcloud by channel

    w, h = calculate_plane(pc)
    ground = np.logical_and(np.matmul(pc[:, :3], np.asarray(w)) + h < 0.5,
                            np.matmul(pc[:, :3], np.asarray(w)) + h > -0.5)
    pc_ground = pc[ground]

    calculated_indicent_angle = np.arccos(np.divide(np.matmul(pc_ground[:, :3], np.asarray(w)),
                                                    np.linalg.norm(pc_ground[:, :3], axis=1) * np.linalg.norm(w)))

    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    sensor_info = Path(__file__).parent.parent.parent.resolve() / 'calib' / '20171102_64E_S3.yaml'

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']

    channels = range(num_channels)
    order = list(range(num_channels))

    if shuffle:
        random.shuffle(order)

    channel_list = [None] * num_channels

    if show_progressbar:

        channel_list[:] = process_map(functools.partial(process_single_channel, root_path, particle_file_prefix,
                                                        orig_pc, beam_divergence, order, channel_infos),
                                      channels, chunksize=4)

    else:

        pool = mp.pool.ThreadPool(mp.cpu_count())

        channel_list[:] = pool.map(functools.partial(process_single_channel, root_path, particle_file_prefix, orig_pc,
                                                     beam_divergence, order, channel_infos), channels)

        pool.close()
        pool.join()

    intensity_diff_sum = 0

    for item in channel_list:

        tmp_intensity_diff_sum, idx, pc = item

        intensity_diff_sum += tmp_intensity_diff_sum

        aug_pc[idx] = pc

    aug_pc[:, 3] = np.round(aug_pc[:, 3])

    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)

    num_removed = np.logical_not(scattered_or_above_threshold).sum()
    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    if only_camera_fov:
        calib = get_calib()

        pts_rect = calib.lidar_to_rect(aug_pc[:, 0:3])
        fov_flag = get_fov_flag(pts_rect, (1024, 1920), calib)

        num_removed += np.logical_not(fov_flag).sum()

        aug_pc = aug_pc[fov_flag]

    stats = num_attenuated, num_removed, avg_intensity_diff

    return  stats, aug_pc


def received_power(CA_P0: float, beta_0: float, ratio: float, r: float, r_j: float, tau_h: float) -> float:

    answer = ((CA_P0 * beta_0 * ratio * xsi(r_j)) / (r_j ** 2)) * np.sin((PI * (r - r_j)) / (c * tau_h)) ** 2

    return answer

def xsi(R: float, R_1: float = 0.9, R_2: float = 1.0) -> float:

    if R <= R_1:    # emitted ligth beam from the tansmitter is not captured by the receiver

        return 0

    elif R >= R_2:  # emitted ligth beam from the tansmitter is fully captured by the receiver

        return 1

    else:           # emitted ligth beam from the tansmitter is partly captured by the receiver

        m = (1 - 0) / (R_2 - R_1)
        b = 0 - (m * R_1)
        y = m * R + b

        return y


if __name__ == '__main__':

    start_angle = np.radians(-22.5)
    field_of_view = np.radians(360)
    angular_resolution = np.radians(0.35)

    n_beams = int(np.floor(field_of_view / angular_resolution))

    right_angles = np.linspace(start=start_angle,
                               stop=start_angle + field_of_view - angular_resolution,
                               num=n_beams, endpoint=True)

    left_angles = np.linspace(start=start_angle + angular_resolution,
                              stop=start_angle + field_of_view,
                              num=n_beams, endpoint=True)

    # Ensure angles are in [0, 2*PI]
    right_angles[right_angles < 0] = right_angles[right_angles < 0] + 2*PI
    left_angles[left_angles < 0] = left_angles[left_angles < 0] + 2*PI

    beam_limit_angles = np.column_stack((right_angles, left_angles))

    ranges = 25 * np.ones(n_beams)

    test_particles = np.array([[10,  2, 1],
                               [15,  4, 2],
                               [ 3,  4, 1],
                               [12, 10, 3],
                               [17, 12, 1],
                               [ 0,  6, 1],
                               [ 7,  0, 1],
                               [18,  1, 1],
                               [ 2,  9, 2],
                               [ 3, -1, 1.1],
                               [30, 12, 3]])

    # result = get_occlusions(beam_limit_angles, ranges, test_particles, float(np.degrees(angular_resolution)))