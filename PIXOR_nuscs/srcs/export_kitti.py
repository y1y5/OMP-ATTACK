# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
- Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
- Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
- This script uses the `train` and `val` splits of nuScenes, whereas standard KITTI has `training` and `testing` splits.

This script includes three main functions:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.
- render_kitti(): Render the annotations of the (generated or real) KITTI dataset.
- kitti_res_to_nuscenes(): Converts a KITTI detection result to the nuScenes detection results format.

To launch these scripts run:
- python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/nusc_kitti
- python export_kitti.py render_kitti --nusc_kitti_dir ~/nusc_kitti --render_2d False
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_dir ~/nusc_kitti
Note: The parameter --render_2d specifies whether to draw 2d or 3d boxes.

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_dir /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""
import json
import os
from typing import List, Dict, Any
import random

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

import argparse

def lidar_nusc_box_to_global(nusc, boxes, sample_token):

    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))

        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        box_list.append(box)
    return box_list


def parse_attacked_label_line(label_line) -> dict:
    """
    Parses single line from label file into a dict. Boxes are in KITTI lidar frame.
    :param label_line: Single line from KittiDB label file.
    :return: Dictionary with all the line details.
    """

    parts = label_line.split(' ')
    output = {
        'score': float(parts[0].strip()),
        'x_lidar': float(parts[1]),
        'y_lidar': float(parts[2]),
        'w': float(parts[3]),
        'l': float(parts[4]),
        'yaw_lidar': float(parts[5]),
    }

    return output


def get_transforms(token: str, root: str='/mnt/data/kitti_nusc') -> dict:

    # split_folder is "split" + "instance_token"
    calib_filename = KittiDB.get_filepath(token, 'calib', root=root)

    lines = [line.rstrip() for line in open(calib_filename)]
    velo_to_cam = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32)
    velo_to_cam.resize((3, 4))

    r0_rect = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32)
    r0_rect.resize((3, 3))
    p_left = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32)
    p_left.resize((3, 4))

    # Merge rectification and projection into one matrix.
    p_combined = np.eye(4)
    p_combined[:3, :3] = r0_rect
    p_combined = np.dot(p_left, p_combined)
    return {
        'velo_to_cam': {
            'R': velo_to_cam[:, :3],
            'T': velo_to_cam[:, 3]
        },
        'r0_rect': r0_rect,
        'p_left': p_left,
        'p_combined': p_combined,
    }


def parse_gt_label_line(label_line) -> dict:

    parts = label_line.split(' ')
    output = {
        'name': parts[0].strip(),
        'xyz_camera': (float(parts[11]), float(parts[12]), float(parts[13])),
        'wlh': (float(parts[9]), float(parts[10]), float(parts[8])),
        'yaw_camera': float(parts[14]),
        'bbox_camera': (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        'truncation': float(parts[1]),
        'occlusion': float(parts[2]),
        'alpha': float(parts[3])
    }

    # Add score if specified
    if len(parts) > 15:
        output['score'] = float(parts[15])
    else:
        output['score'] = np.nan

    return output


def get_gt_box(token: str, root: str ) -> List[Box]:
    
    # Get transforms matrix for this sample
    transforms = get_transforms(token, root=root)

    with open(KittiDB.get_filepath(token, 'label_2', root=root), 'r') as f:
        line = f.readline()

    # Parse this line into box information.
    parsed_line = parse_gt_label_line(line)

    center = parsed_line['xyz_camera']
    wlh = parsed_line['wlh']
    yaw_camera = parsed_line['yaw_camera']
    name = parsed_line['name']
    score = parsed_line['score']

    # 1: Create box in Box coordinate system with center at origin.
    # The second quaternion in yaw_box transforms the coordinate frame from the object frame
    # to KITTI camera frame. The equivalent cannot be naively done afterwards, as it's a rotation
    # around the local object coordinate frame, rather than the camera frame.
    quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(axis=(1, 0, 0), angle=np.pi/2)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

    # 2: Translate: KITTI defines the box center as the bottom center of the vehicle. We use true center,
    # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(center + np.array([0, -wlh[2] / 2, 0]))

    # 3: Transform to KITTI LIDAR coord system. First transform from rectified camera to camera, then
    # camera to KITTI lidar.
    box.rotate(Quaternion(matrix=transforms['r0_rect']).inverse)
    box.translate(-transforms['velo_to_cam']['T'])
    box.rotate(Quaternion(matrix=transforms['velo_to_cam']['R']).inverse)
    # print(box)

    return box


def get_attacked_boxes(token: str,
                       gt_box: Box,
                       root: str,
                       max_dist: float = None) -> List[Box]:

    boxes = []
    split_folder, inst_sample_token = token.split('_')

    with open(os.path.join(root, split_folder, 'label_2_attacked', inst_sample_token+'.txt'), 'r') as f:
        for line in f:
            # Parse this line into box information.
            parsed_line = parse_attacked_label_line(line)

            gt_z = gt_box.center[2]
            gt_h = gt_box.wlh[2]

            score = parsed_line['score']
            x = parsed_line['x_lidar']
            y = parsed_line['y_lidar']
            wlh = (parsed_line['w'], parsed_line['l'], gt_h)
            yaw_lidar = parsed_line['yaw_lidar']
            quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_lidar)

            box = Box([x, y, gt_z], wlh, quat_box, name='car')

            # 4: Transform to nuScenes LIDAR coord system.
            kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
            box.rotate(kitti_to_nu_lidar)

            # Set score or NaN.
            box.score = score

            # Set dummy velocity.
            box.velocity = np.array((0.0, 0.0, 0.0))

            # Optional: Filter by max_dist
            if max_dist is not None:
                dist = np.sqrt(np.sum(box.center[:2] ** 2))
                if dist > max_dist:
                    continue

            boxes.append(box)

    return boxes

class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = None,
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 10,
                 nusc_version: str = 'v1.0-trainval',
                 split: str = 'train',
                 nusc: NuScenes = None):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = nusc

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, self.split, 'label_2')  # ~/nusc_kitti/train/label_2
        calib_folder = os.path.join(self.nusc_kitti_dir, self.split, 'calib')  # ~/nusc_kitti/train/calib
        image_folder = os.path.join(self.nusc_kitti_dir, self.split, 'image_2')  # ~/nusc_kitti/train/image_2
        lidar_folder = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne')  # ~/nusc_kitti/train/velodyne
        for folder in [label_folder, calib_folder, image_folder, lidar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        tokens = []
        for sample_token in sample_tokens:

            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            # token = '%06d' % token_idx # Alternative to use KITTI names.
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, sample_token + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, sample_token + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, sample_token + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write label file.
            label_path = os.path.join(label_folder, sample_token + '.txt')
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                continue
            else:
                print('Writing file: %s' % label_path)
            with open(label_path, "w") as label_file:
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])
                    box_lidar_nusc = box_lidar_nusc[0]

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    occluded = 0

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])

                    # Skip categories that are not part of the nuScenes detection challenge.
                    if detection_name is None:
                        continue

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        continue

                    # Set dummy score so we can use this file as result.
                    box_cam_kitti.score = 0

                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                   truncation=truncated, occlusion=occluded)

                    # Write to disk.
                    label_file.write(output + '\n')

    def render_kitti(self, render_2d: bool) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split,))

        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        for token in kitti.tokens[:self.image_count]:
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                print('Rendering file to disk: %s' % out_path)
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                plt.close()  # Close the windows to avoid a warning of too many open windows.

    def kitti_res_to_nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split, ))

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        for sample_token in sample_tokens:
            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, sample_token)
            boxes = kitti.get_boxes(token=kitti_token)

            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

            # Store all results for this image.
            results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


    def target_nuscenes_to_kitti(self, scene_name: str, target_instance_token: str, target_sample_token_list: List[str]) -> None:
            """
            Converts nuScenes GT annotations to KITTI format.
            """
            kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
            kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
            imsize = (1600, 900)

            token_idx = 0  # Start tokens from 0.

            # Get assignment of scenes to splits.
            # split_logs = create_splits_logs(self.split, self.nusc)

            # Create output folders.
            folder_name = '{}-{}'.format(scene_name, target_instance_token)
            label_folder = os.path.join(self.nusc_kitti_dir, folder_name, 'label_2')
            calib_folder = os.path.join(self.nusc_kitti_dir, folder_name, 'calib')
            image_folder = os.path.join(self.nusc_kitti_dir, folder_name, 'image_2')
            lidar_folder = os.path.join(self.nusc_kitti_dir, folder_name, 'velodyne')
            plot_folder = os.path.join(self.nusc_kitti_dir, folder_name, 'plots')
            for folder in [label_folder, calib_folder, image_folder, lidar_folder, plot_folder]:
                if not os.path.isdir(folder):
                    os.makedirs(folder)

            # # Use only the samples from the current split.
            # sample_tokens = self._split_to_samples(split_logs)
            # sample_tokens = sample_tokens[:self.image_count]
            sample_tokens = target_sample_token_list

            # select targeted token for attack
            tokens = []
            for sample_token in sample_tokens:

                # Get sample data.
                sample = self.nusc.get('sample', sample_token)
                sample_annotation_tokens = sample['anns']
                
                # find anno token for targeted instance
                target_annotation_tokens = []
                for t in sample_annotation_tokens:
                    sample_anno = self.nusc.get('sample_annotation', t)
                    if sample_anno['instance_token'] == target_instance_token:
                        target_annotation_tokens.append(t)
                        self.nusc.render_annotation(t, out_path='{}/{}.png'.format(plot_folder, sample_token))
                        break
                sample_annotation_tokens = target_annotation_tokens
            
                cam_front_token = sample['data'][self.cam_name]
                lidar_token = sample['data'][self.lidar_name]

                # Retrieve sensor records.
                sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                            inverse=False)
                ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                            inverse=True)
                velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

                # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
                velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
                r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

                # Projection matrix.
                p_left_kitti = np.zeros((3, 4))
                p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

                # Check that the rotation has the same format as in KITTI.
                assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
                assert (velo_to_cam_trans[1:3] < 0).all()

                # Retrieve the token from the lidar.
                # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
                # not the camera.
                filename_cam_full = sd_record_cam['filename']
                filename_lid_full = sd_record_lid['filename']
                # token = '%06d' % token_idx # Alternative to use KITTI names.
                token_idx += 1

                # Convert image (jpg to png).
                src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
                dst_im_path = os.path.join(image_folder, sample_token + '.png')
                if not os.path.exists(dst_im_path):
                    im = Image.open(src_im_path)
                    im.save(dst_im_path, "PNG")

                # Convert lidar.
                # Note that we are only using a single sweep, instead of the commonly used n sweeps.
                src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
                dst_lid_path = os.path.join(lidar_folder, sample_token + '.bin')
                assert not dst_lid_path.endswith('.pcd.bin')
                pcl = LidarPointCloud.from_file(src_lid_path)
                pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
                with open(dst_lid_path, "w") as lid_file:
                    pcl.points.T.tofile(lid_file)

                # Add to tokens.
                tokens.append(sample_token)

                # Create calibration file.
                kitti_transforms = dict()
                kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
                kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
                kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
                calib_path = os.path.join(calib_folder, sample_token + '.txt')
                with open(calib_path, "w") as calib_file:
                    for (key, val) in kitti_transforms.items():
                        val = val.flatten()
                        val_str = '%.12e' % val[0]
                        for v in val[1:]:
                            val_str += ' %.12e' % v
                        calib_file.write('%s: %s\n' % (key, val_str))

                # Write label file.
                label_path = os.path.join(label_folder, sample_token + '.txt')
                if os.path.exists(label_path):
                    print('Skipping existing file: %s' % label_path)
                    continue
                else:
                    print('Writing file: %s' % label_path)
                with open(label_path, "w") as label_file:
                    for sample_annotation_token in sample_annotation_tokens:
                        sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                        # Get box in LIDAR frame.
                        _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                        selected_anntokens=[sample_annotation_token])
                        box_lidar_nusc = box_lidar_nusc[0]

                        # Truncated: Set all objects to 0 which means untruncated.
                        truncated = 0.0

                        # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                        occluded = 0

                        # Convert nuScenes category to nuScenes detection challenge category.
                        detection_name = category_to_detection_name(sample_annotation['category_name'])

                        # Skip categories that are not part of the nuScenes detection challenge.
                        if detection_name is None:
                            continue

                        # Convert from nuScenes to KITTI box format.
                        box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                            box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                        # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                        bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                        # if bbox_2d is None:
                        #     continue
                        if bbox_2d is None:  # keep boxes that are not in the image
                            bbox_2d = (-10, -10, -10, -10)

                        # Set dummy score so we can use this file as result.
                        box_cam_kitti.score = 0

                        # Convert box to output string format.
                        output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                    truncation=truncated, occlusion=occluded)

                        # Write to disk.
                        label_file.write(output + '\n')


    def attack_kitti_to_nusc(self, inst_sample_token_list: List[str] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param inst_sample_token_list: instance token - sample token list
        """

        # Init.
        results = {}

        for inst_sample_token in inst_sample_token_list:

            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, inst_sample_token)

            # to nuScenes LIDAR coord system
            gt_box_lidar = get_gt_box(token=kitti_token, root=self.nusc_kitti_dir)  # get gt z and h to assign to predicted boxes
            attacked_boxes_lidar = get_attacked_boxes(token=kitti_token, \
                                                      gt_box=gt_box_lidar, root=self.nusc_kitti_dir) # label_2_attacked

            # Convert nuScenes box in lidar frame to global frame
            sample_token = inst_sample_token.split('-')[-1]
            boxes = lidar_nusc_box_to_global(nusc=self.nusc, boxes=attacked_boxes_lidar, sample_token=sample_token)

            # # Convert KITTI boxes to nuScenes detection challenge result format.
            inst_sample_results = [self._box_to_sample_result(inst_sample_token, box) for box in boxes]

            # Store all results for this image.
            results[inst_sample_token] = inst_sample_results

        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chooose scene from nuScenes transfer to KITTI format')
    parser.add_argument('--root_dir', type=str, default='mnt/dataset/nuScenes/results')
    parser.add_argument('--target_scene_instance_file', type=str, default='mnt/dataset/nuScenes/results/attack_pred/scene_inst_tokens.txt') # todo: need to specify the scene and instance
    parser.add_argument('--nusc_datadir', type=str, default='mnt/dataset/nuScenes/trainval')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    args = parser.parse_args()

    output_dir = args.root_dir
    ### load scene instance pairs ###
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_datadir, verbose=True)
    #scene_name = "scene-0103"
    target_scene_inst = {}
    with open(args.target_scene_instance_file, 'r') as f:
        for line in f.readlines():
            t = line.split('-')
            scene = '-'.join(t[:2])
            instance = t[-1].strip()
            if scene not in target_scene_inst:
                target_scene_inst[scene] = []
            target_scene_inst[scene].append(instance)

    for scene_name in target_scene_inst:
        # if scene_name != "scene-0001":
        #     # todo: specify the scene
        #     continue
        target_instance_tokens = target_scene_inst[scene_name]
        for target_instance_token in target_instance_tokens:
            instance = nusc.get('instance', target_instance_token)
            first_ann_token = instance['first_annotation_token']

            target_sample_token_list = []
            current_ann_token = first_ann_token
            sample_token = nusc.get('sample_annotation', current_ann_token)['sample_token']
            while sample_token:
                target_sample_token_list.append(sample_token)
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']

            kitti_converter = KittiConverter(nusc=nusc, nusc_kitti_dir=output_dir, image_count=10000)
            kitti_converter.target_nuscenes_to_kitti(scene_name=scene_name, target_instance_token=target_instance_token, target_sample_token_list=target_sample_token_list)
            output_file_path = os.path.join(output_dir, scene_name + '-' + target_instance_token, 'target_sample_tokens.txt')
            with open(output_file_path, 'w') as f:
                for token in target_sample_token_list:
                    f.write(f"{token}\n")

            # config.yaml
            import shutil
            source_config_path = 'mnt/dataset/nuScenes/results/config.yaml'
            target_config_path = os.path.join(output_dir, scene_name + '-' + target_instance_token, 'config.yaml')
            shutil.copy(source_config_path, target_config_path)



