from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import glob
import json
import imageio
import cv2
import re
from torchvision.transforms import ColorJitter, GaussianBlur

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

from enum import Enum

IGNORE_ANIMALS = [
    # "bear.json",
    # "camel.json",
    "cat_jump.json"
    # "cows.json",
    # "dog.json",
    # "dog-agility.json",
    # "horsejump-high.json",
    # "horsejump-low.json",
    # "impala0.json",
    # "rs_dog.json"
    "tiger.json"
]


class SMALJointCatalog(Enum):
    # body_0 = 0
    # body_1 = 1
    # body_2 = 2
    # body_3 = 3
    # body_4 = 4
    # body_5 = 5
    # body_6 = 6
    # upper_right_0 = 7
    upper_right_1 = 8
    upper_right_2 = 9
    upper_right_3 = 10
    # upper_left_0 = 11
    upper_left_1 = 12
    upper_left_2 = 13
    upper_left_3 = 14
    neck_lower = 15
    # neck_upper = 16
    # lower_right_0 = 17
    lower_right_1 = 18
    lower_right_2 = 19
    lower_right_3 = 20
    # lower_left_0 = 21
    lower_left_1 = 22
    lower_left_2 = 23
    lower_left_3 = 24
    tail_0 = 25
    # tail_1 = 26
    # tail_2 = 27
    tail_3 = 28
    # tail_4 = 29
    # tail_5 = 30
    tail_6 = 31
    jaw = 32
    nose = 33 # ADDED JOINT FOR VERTEX 1863
    # chin = 34 # ADDED JOINT FOR VERTEX 26
    right_ear = 35 # ADDED JOINT FOR VERTEX 149
    left_ear = 36 # ADDED JOINT FOR VERTEX 2124

class SMALJointInfo():
    def __init__(self):
        # These are the 
        self.annotated_classes = np.array([
            8, 9, 10, # upper_right
            12, 13, 14, # upper_left
            15, # neck
            18, 19, 20, # lower_right
            22, 23, 24, # lower_left
            25, 28, 31, # tail
            32, 33, # head
            35, # right_ear
            36]) # left_ear

        self.annotated_markers = np.array([
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS])

        self.joint_regions = np.array([ 
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 
            2, 2, 2, 2, 
            3, 3, 
            4, 4, 4, 4, 
            5, 5, 5, 5, 
            6, 6, 6, 6, 6, 6, 6,
            7, 7, 7,
            8, 
            9])

        self.annotated_joint_region = self.joint_regions[self.annotated_classes]
        self.region_colors = np.array([
            [250, 190, 190], # body, light pink
            [60, 180, 75], # upper_right, green
            [230, 25, 75], # upper_left, red
            [128, 0, 0], # neck, maroon
            [0, 130, 200], # lower_right, blue
            [255, 255, 25], # lower_left, yellow
            [240, 50, 230], # tail, majenta
            [245, 130, 48], # jaw / nose / chin, orange
            [29, 98, 115], # right_ear, turquoise
            [255, 153, 204]]) # left_ear, pink
        
        self.joint_colors = np.array(self.region_colors)[self.annotated_joint_region]


class BADJAData():
    def __init__(self, data_root, complete=False):
        annotations_path = os.path.join(data_root, "joint_annotations")
        print('annotations_path', annotations_path)

        self.animal_dict = {}
        self.animal_count = 0
        self.smal_joint_info = SMALJointInfo()
        for animal_id, animal_json in enumerate(sorted(os.listdir(annotations_path))):
            if animal_json not in IGNORE_ANIMALS:
                json_path = os.path.join(annotations_path, animal_json)
                with open(json_path) as json_data:
                    animal_joint_data = json.load(json_data)

                filenames = []
                segnames = []
                joints = []
                visible = []

                print('number of annotated frames', len(animal_joint_data))

                first_path = animal_joint_data[0]['segmentation_path']
                last_path = animal_joint_data[-1]['segmentation_path']
                first_frame = first_path.split('/')[-1]
                last_frame = last_path.split('/')[-1]

                if not 'extra_videos' in first_path:


                    # if 'extra_videos' in first_path:
                    #     animal = first_path.split('/')[1]
                    #     base = 
                    # else:
                    animal = first_path.split('/')[-2]

                    # print('animal', animal)
                    # import ipdb; ipdb.set_trace()

                    # folder = first_path[:-len(first_frame)]
                    first_frame_int = int(first_frame.split('.')[0])
                    last_frame_int = int(last_frame.split('.')[0])

                    for fr in range(first_frame_int, last_frame_int+1):
                        ref_file_name = os.path.join(data_root, 'DAVIS/JPEGImages/Full-Resolution/%s/%05d.jpg' % (animal, fr))
                        ref_seg_name = os.path.join(data_root, 'DAVIS/Annotations/Full-Resolution/%s/%05d.png' % (animal, fr))
                        # print('ref_file_name', ref_file_name)

                        foundit = False
                        for ind, image_annotation in enumerate(animal_joint_data):
                            file_name = os.path.join(data_root, image_annotation['image_path'])
                            seg_name = os.path.join(data_root, image_annotation['segmentation_path'])

                            # print('file_name', file_name)

                            if file_name == ref_file_name:
                                foundit = True
                                label_ind = ind

                        if foundit:
                            image_annotation = animal_joint_data[label_ind]
                            file_name = os.path.join(data_root, image_annotation['image_path'])
                            seg_name = os.path.join(data_root, image_annotation['segmentation_path'])
                            joint = np.array(image_annotation['joints'])
                            vis = np.array(image_annotation['visibility'])
                        else:
                            file_name = ref_file_name
                            seg_name = ref_seg_name
                            # seg_name = None
                            joint = None
                            vis = None

                        filenames.append(file_name)
                        segnames.append(seg_name)
                        joints.append(joint)
                        visible.append(vis)
                    
                # print('filenames', filenames)
                # print('segnames', segnames)
                if len(filenames):
                    # print('adding this (original) id:', animal_id)
                    self.animal_dict[self.animal_count] = (filenames, segnames, joints, visible)
                    self.animal_count += 1
        print ("Loaded BADJA dataset")

    def get_loader(self):
        for idx in range(int(1e6)):
            animal_id = np.random.choice(len(self.animal_dict.keys()))
            # print('choosing animal_id', animal_id)
            filenames, segnames, joints, visible = self.animal_dict[animal_id]
            # print('filenames', filenames)

            image_id = np.random.randint(0, len(filenames))

            seg_file = segnames[image_id]
            image_file = filenames[image_id]
            
            joints = joints[image_id].copy()
            joints = joints[self.smal_joint_info.annotated_classes]
            visible = visible[image_id][self.smal_joint_info.annotated_classes]

            rgb_img = imageio.imread(image_file)#, mode='RGB')
            sil_img = imageio.imread(seg_file)#, mode='RGB')

            rgb_h, rgb_w, _ = rgb_img.shape
            sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

            yield rgb_img, sil_img, joints, visible, image_file

    def get_video(self, animal_id):
        # print('choosing animal_id', animal_id)
        filenames, segnames, joint, visible = self.animal_dict[animal_id]
        # print('filenames', filenames)

        rgbs = []
        segs = []
        joints = []
        visibles = []

        for s in range(len(filenames)):
            image_file = filenames[s]
            rgb_img = imageio.imread(image_file)#, mode='RGB')
            rgb_h, rgb_w, _ = rgb_img.shape
            
            seg_file = segnames[s]
            sil_img = imageio.imread(seg_file)#, mode='RGB')
            sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)
            
            jo = joint[s]

            # print('image_file', image_file)
            # print('seg_file', seg_file)
            if jo is not None:
                joi = joint[s].copy()
                joi = joi[self.smal_joint_info.annotated_classes]
                vis = visible[s][self.smal_joint_info.annotated_classes]
            else:
                joi = None
                vis = None

            rgbs.append(rgb_img)
            segs.append(sil_img)
            joints.append(joi)
            visibles.append(vis)

        return rgbs, segs, joints, visibles, filenames[0]



class BadjaDataset(torch.utils.data.Dataset):
    def __init__(self, data_root='../badja'):

        self.data_root = data_root
        self.badja_data = BADJAData(data_root)
        print('found %d unique videos in %s' % (self.badja_data.animal_count, self.data_root))
        
    def __getitem__(self, index):
        
        rgbs, segs, joints, visibles, file0 = self.badja_data.get_video(index)
        S = len(rgbs)
        H, W, C = rgbs[0].shape
        H, W, mystery = segs[0].shape
        # print('joints[0]', joints[0].shape)
        N, D = joints[0].shape

        # let's eliminate the Nones
        # note the first one is guaranteed present
        for s in range(1,S):
            if joints[s] is None:
                # segs[s] = np.zeros_like(segs[0])
                joints[s] = np.zeros_like(joints[0])
                visibles[s] = np.zeros_like(visibles[0])

        # eliminate the mystery dim
        segs = [seg[:,:,0] for seg in segs]
        
        # print('rgb', rgbs[0].shape)
        # print('seg', segs[0].shape)

        rgbs = np.stack(rgbs, 0)
        segs = np.stack(segs, 0)
        trajs = np.stack(joints, 0)
        visibles = np.stack(visibles, 0)
        # print('rgbs', rgbs.shape)
        # print('segs', segs.shape)
        # print('trajs', trajs.shape)
        # print('visibles', visibles.shape)
        
        rgbs = torch.from_numpy(rgbs).reshape(S, H, W, 3).permute(0,3,1,2)
        segs = torch.from_numpy(segs).reshape(S, 1, H, W)
        trajs = torch.from_numpy(trajs).reshape(S, N, 2)
        visibles = torch.from_numpy(visibles).reshape(S, N)

        # apparently the coords are in yx order
        trajs = torch.flip(trajs, [2])
        
        # trajs = torch.cat([trajs[:,:,1],trajs[:,:,0]], dim=2)
        
        # invs = torch.from_numpy(np.stack(invs, 0)).reshape(S, H, W, 1).permute(0,3,1,2)
        # flows = torch.from_numpy(np.stack(flows, 0)).reshape(S, H, W, 2).permute(0,3,1,2)

        return_dict = {
            'file0': file0,
            'rgbs': rgbs,
            'segs': segs,
            'trajs': trajs,
            'visibles': visibles,
        }
        return return_dict
        
    def __len__(self):
        # return 10
        # return len(self.rgb_paths)
        return self.badja_data.animal_count


    
