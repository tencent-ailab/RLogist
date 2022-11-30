import os
import math
from typing import Optional
import numpy as np
import pandas as pd
import gym
from gym import spaces
from create_patches import seg_and_patch, patching, segment
from wsi_core.WholeSlideImage import WholeSlideImage
from utils.feature_updater import f_local, f_global


class WSIObservationEnv(gym.Env):
    """
    ### Description
    Environment for RLogist with discrete action space and feature update mechanism.

    ### Observation Space
    The observation is a `ndarray` with shape `(REGION_NUM, EMBEDDING_LENGTH)` where the elements correspond to the following:
    | Num | Observation            | Min  | Max |
    |-----|------------------------|------|-----|
    | 0   | Index of the region    | 0    | REGION_NUM-1 |
    | 1   | Scanning-level feature | -Inf | Inf |

    ### Action Space
    There is 1 discrete deterministic actions:
    | Num | Observation                                     | Value    |
    |-----|-------------------------------------------------|----------|
    | 0   | the index of target region for analyze in depth | [0, REGION_NUM-1] |

    ### Transition Dynamics:
    Given an action, the WSIObservationEnv follows the following transition dynamics:
    Update the scanning-level feature of all unobserved regions with f_local and f_global.

    ### Reward:
    The goal is to predict the slide-level label, as such the agent assigned with a reward of 1 for the
    right classification result.

    ### Starting State
    The scanning-level feature is extracted with pretrained models i.e. ResNet50

    ### Episode End
    The episode ends if the length of the episode reaches MAX_LENGTH.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, WSI_object: WholeSlideImage, scanning_level, deep_level):

        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
        vis_params = {'vis_level': -1, 'line_thickness': 250}
        patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
        patch_size = 256
        step_size = 256
        region_num = 200

        current_vis_params = vis_params.copy()
        current_filter_params = filter_params.copy()
        current_seg_params = seg_params.copy()
        current_patch_params = patch_params.copy()

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)


        patch_time_elapsed = -1
        current_patch_params.update({'scanning_level': scanning_level, 'patch_size': patch_size, 'step_size': step_size})
        file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(region_num, 1024, dtype=np.float32)
