# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import cv2
import torch
import copy
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_bool
from utils import *

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_point_grid,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.87,
        stability_score_thresh: float = 0.85,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.5,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 25,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.points_per_side = points_per_side
            #self.point_grids = build_point_grid(points_per_side)
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: np.ndarray,ref_bbox) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data, mask_size = self._generate_masks(image,ref_bbox)
        mask_area = mask_size*mask_size
        self.min_mask_region_area = mask_area/4

        curr_anns = []
        if len(mask_data._stats)<1:
            return curr_anns

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        
        for idx in range(len(mask_data["segmentations"])):
            mask = mask_data["segmentations"][idx]
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns
    
    def _generate_similarity(self, image, ref_bbox, iter=1):
        img_size = image.shape[:2]
        ref_bbox = torch.tensor(ref_bbox, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(ref_bbox, img_size)
        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
            )
        
        masks_cpu = masks.cpu()
        mask_size = [math.sqrt(np.sum(mask.float().numpy())) for mask in masks_cpu]
        mask_size = np.array(mask_size).mean(0)
        
        feat = self.predictor.get_image_embedding().squeeze()
        ref_feat = feat.permute(1, 2, 0)
        C, h, w = feat.shape
        test_feat = feat / feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        low_res_masks = F.interpolate(low_res_masks, size=ref_feat.shape[0: 2], mode='bilinear', align_corners=False)
        low_res_masks = low_res_masks.flatten(2, 3)
        masks_low_res = (low_res_masks > self.predictor.model.mask_threshold).float()
        topk_idx = torch.topk(low_res_masks, 1)[1]
        masks_low_res.scatter_(2, topk_idx, 1.0)
        target_embedding = []
        sim = []
        for i, ref_mask in enumerate(masks_low_res.cpu()):
            ref_mask = ref_mask.squeeze().reshape(ref_feat.shape[0: 2])

            # Target feature extraction
            target_feat = ref_feat[ref_mask > 0]
            if target_feat.shape[0]>0:
                target_embedding_ = target_feat.mean(0).unsqueeze(0)
                target_feat = target_embedding_ / target_embedding_.norm(dim=-1, keepdim=True)
                target_embedding_ = target_embedding_.unsqueeze(0)
                target_embedding.append(target_embedding_)

                sim_ = target_feat @ test_feat
                sim_ = sim_.reshape(1, 1, h, w)
                sim_ = F.interpolate(sim_, scale_factor=4, mode="bilinear")
                sim_ = self.predictor.model.postprocess_masks(
                                sim_,
                                input_size=self.predictor.input_size,
                                original_size=self.predictor.original_size).squeeze()#"""
                sim_ = sim_.cpu().numpy()
                sim.append(sim_)

        sim = np.array(sim).mean(0)
        target_embedding = torch.mean(torch.concat(target_embedding, dim=0), dim=0, keepdim=True)

        return sim, target_embedding, mask_size


    def _generate_masks(self, image: np.ndarray,ref_bbox) -> MaskData:
        orig_size = image.shape[:2]
        self.predictor.set_image(image)

        #Computing similarity map
        sim, target_embedding, mask_size = self._generate_similarity(image, ref_bbox, iter=1)

        #refine similarity map
        ref_bbox_new = get_boxes_from_sim(sim)
        sim, target_embedding, mask_size = self._generate_similarity(image, ref_bbox_new, iter=2)

        target_size = self.predictor.transform.get_preprocess_shape(sim.shape[0], sim.shape[1], self.predictor.transform.target_length)
        sim_map = resize(sim,target_size,preserve_range=True)

        #"""
        T = np.max(sim_map)/1.3
        sim_map[sim_map<T]=0
        sim_map[sim_map>=T]=1 #"""

        """
        sim_map = cv2.convertScaleAbs(sim_map*255)
        th,sim_map = cv2.threshold(sim_map,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#"""

        points_per_side = (self.points_per_side//(1.2*mask_size)+1)*self.points_per_side
        self.point_grids = build_point_grid(int(points_per_side))
        
        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids * points_scale

        point_mask = np.zeros(orig_size, dtype=bool)

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):

            #Applying for segment prior to remove redundant points in the current batch
            points_new = []
            for point_i in points:
                point_i = point_i.astype(int)
                if not point_mask[point_i[1],point_i[0]]:
                    points_new.append(point_i)

            if len(points_new)>0:
                batch_data, point_mask = self._process_batch(np.array(points_new), orig_size, sim_map, target_embedding, point_mask)
                if len(batch_data._stats)>0:
                    data.cat(batch_data)
                del batch_data

        self.predictor.reset_image()

        if len(data._stats)>0:
            # Remove duplicates within this crop.
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data, mask_size

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        sim_map,
        target_embedding,
        point_mask,
    ):
        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size).astype(int)
        # Applying for similarity prior to label negative and positive points
        transformed_labels = sim_map[[transformed_points[:,1],transformed_points[:,0]]]
        # The batch would be passed if almost every points in the batch is negative
        if np.sum(transformed_labels)<(transformed_points.shape[0]/16):
            return MaskData(), point_mask
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.as_tensor(transformed_labels, dtype=torch.int, device=in_points.device)

        masks, iou_preds, logits = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            return_logits=True,
            target_embedding=target_embedding  # Semantic prior
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        ) 
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Compress to RLE
        data["rles"] = mask_to_rle_pytorch(data["masks"])

        # Maintain an overall segment map that contains all the segmented regions up to the current batch.
        for mask_i in data["masks"].cpu():
            point_mask = np.logical_or(point_mask, mask_i)

        del data["masks"]

        return data, point_mask

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data