#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
import contextlib
import logging
import json
import os
from dqrf.utils.utils import ImageMeta
"""
This file contains functions to parse Crowdhuman-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger("detectron2.data.datasets.coco")

class get_crowdhuman_val_dicts(object):
    def __init__(self, json_file, image_root):
        """
        Note that crowdhuman box format is XYXY
        :param json_file: str, full path to the json file in CH instances annotation format.
        :param image_root: str or path-like, the directory where the images in this json file exists
        :return: list[dict] each dict contains file_name, image_id, height, width,
        """
        self.json_file = json_file
        self.image_root = image_root

    def __call__(self):
        timer = Timer()
        json_file = PathManager.get_local_path(self.json_file)
        with open(json_file, 'r') as file:
            # imgs_anns = json.load(file)
            imgs_anns = file.readlines()
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        logger.info("Loaded {} images in CrowdHuman format from {}".format(
            len(imgs_anns), json_file))

        dataset_dicts = []
        # aspect_ratios = []

        for idx, ann in enumerate(imgs_anns):
            anno = json.loads(ann)
            record = {}

            record["image_id"] = idx + 1
            record["ID"] = anno["ID"]
            record["file_name"] = os.path.join(self.image_root, anno["ID"] + ".jpg")

            objs = []
            for gt_box in anno["gtboxes"]:
                if gt_box["fbox"][2] < 0 or gt_box["fbox"][3] < 0:
                    continue
                obj = {}
                obj["bbox"] = gt_box["fbox"]
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) != 0:
                    obj["category_id"] = -1
                else:
                    obj["category_id"] = 0

                vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(
                    (gt_box["fbox"][2] * gt_box["fbox"][3])
                )
                obj["vis_ratio"] = vis_ratio

                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

