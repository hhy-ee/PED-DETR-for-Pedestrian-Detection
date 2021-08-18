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

class get_crowdhuman_dicts(object):
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
            v = json.loads(ann)
            record = {}
            if idx == 1171:
                a =1 
            filename = v["ID"] + '.jpg'
            # NOTE when filename starts with '/', it is an absolute filename thus os.path.join doesn't work
            if filename.startswith('/'):
                filename = os.path.normpath(self.image_root + filename)
            else:
                filename = os.path.join(self.image_root, filename)
            # height, width = v["image_height"], v["image_width"]

            record["file_name"] = filename
            record["image_id"] = idx
            # record["height"] = height
            # record["width"] = width

            objs = []
            for anno in v.get('gtboxes', []):
                fx1, fy1, fw, fh = anno['fbox']
                fx2 = fx1 + fw
                fy2 = fy1 + fh
                vx1, vy1, vw, vh = anno['vbox']
                vx2 = vx1 + vw
                vy2 = vy1 + vh
                fbox = [fx1, fy1, fx2, fy2]
                vbox = [vx1, vy1, vx2, vy2]
                is_ignored = anno['head_attr'].get('ignore', False) == 1
                
                if anno['tag'] == 'person':
                    obj = {
                        "category_id": 1,
                        "bbox": fbox,
                        "vbbox": vbox,
                        "is_ignored": is_ignored,
                        'area': fw * fh,
                        # 'bbox_mode': BoxMode.XYXY_ABS
                    }
                    objs.append(obj)
            # ratio = 1.0 * (height + 1) / (width + 1) # do something with ratio ?
            record["annotations"] = objs
            # dataset_dicts.append(record) # to print class histogram
            dataset_dicts.append(ImageMeta.encode(record)) #this saves up to x2 memory when serializing the data
            # aspect_ratios.append(ratio)
        return dataset_dicts


    # def __call__(self):
    #     timer = Timer()
    #     json_file = PathManager.get_local_path(self.json_file)
    #     with open(json_file, 'r') as file:
    #         # imgs_anns = json.load(file)
    #         imgs_anns = file.readlines()
    #     if timer.seconds() > 1:
    #         logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    #     logger.info("Loaded {} images in CrowdHuman format from {}".format(
    #         len(imgs_anns), json_file))

    #     annos = [json.loads(line.strip()) for line in imgs_anns]

    #     dataset_dicts = []
    #     for img_id, anno in enumerate(annos):
    #         record = {}
    #         record["image_id"] = img_id
    #         record["file_name"] = os.path.join(self.image_root, anno["ID"] + ".jpg")

    #         objs = []
    #         for gt_box in anno["gtboxes"]:
    #             if gt_box["fbox"][2] < 0 or gt_box["fbox"][3] < 0:
    #                 continue
    #             obj = {}
    #             obj["bbox"] = gt_box["fbox"]
    #             obj["vbbox"] = gt_box["vbox"]
    #             if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) != 0:
    #                 obj["category_id"] = 0
    #             else:
    #                 obj["category_id"] = 1

    #             vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(
    #                 (gt_box["fbox"][2] * gt_box["fbox"][3])
    #             )
    #             obj["vis_ratio"] = vis_ratio

    #             objs.append(obj)
    #         record["annotations"] = objs
    #         dataset_dicts.append(ImageMeta.encode(record))
    #     return dataset_dicts


