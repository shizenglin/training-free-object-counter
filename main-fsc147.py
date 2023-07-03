import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
import time
import cv2
from tqdm import tqdm
from os.path import exists,join
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import clip
from shi_segment_anything import sam_model_registry, SamPredictor
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from utils import *

parser = argparse.ArgumentParser(description="Counting with SAM")
parser.add_argument("-dp", "--data_path", type=str, default='./dataset/FSC147_384_V2/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave/FSC147", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point", "text"], help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str,default='cuda:0', help="device")
args = parser.parse_args()

if __name__=="__main__": 

    data_path = args.data_path
    anno_file = data_path + 'annotation_FSC147_384.json'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images'

    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir+'/logs')
    
    if not exists(args.output_dir+'/%s'%args.test_split):
        os.mkdir(args.output_dir+'/%s'%args.test_split)

    if not exists(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type)):
        os.mkdir(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type))
    
    log_file = open(args.output_dir+'/logs/log-%s-%s.txt'%(args.test_split,args.prompt_type), "w")    

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)
    
    if args.prompt_type=='text':
        from shi_segment_anything.automatic_mask_generator_text import SamAutomaticMaskGenerator

        with open(data_path+'ImageClasses_FSC147.txt') as f:
            class_lines = f.readlines()

        class_dict = {}
        for cline in class_lines:
            strings = cline.strip().split('\t')
            class_dict[strings[0]] = strings[1]
        
        clip_model, _ = clip.load("CS-ViT-B/16", device=args.device)
        clip_model.eval()

    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    MAE = 0
    RMSE = 0
    NAE = 0
    SRE = 0
    im_ids = data_split[args.test_split]
    for i,im_id in tqdm(enumerate(im_ids)):
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.prompt_type=='text':
            cls_name = class_dict[im_id]
            input_prompt = get_clip_bboxs(clip_model,image,cls_name,args.device)
        else:         
            input_prompt = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                if args.prompt_type=='box':
                    input_prompt.append([x1, y1, x2, y2])
                elif args.prompt_type=='point':
                    input_prompt.append([(x1+x2)//2, (y1+y2)//2])
        
        masks = mask_generator.generate(image, input_prompt)

        gt_cnt = dots.shape[0]
        pred_cnt = len(masks)

        print(pred_cnt, gt_cnt, abs(pred_cnt-gt_cnt))
        log_file.write("%d: %d,%d,%d\n"%(i, pred_cnt, gt_cnt,abs(pred_cnt-gt_cnt)))
        log_file.flush()

        err = abs(gt_cnt - pred_cnt)
        MAE = MAE + err
        RMSE = RMSE + err**2
        NAE = NAE+err/gt_cnt
        SRE = SRE+err**2/gt_cnt

        #Mask visualization
        """
        fig = plt.figure()
        plt.axis('off')
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.imshow(image)
        show_anns(masks, plt.gca())
        plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        plt.close()#"""

    MAE = MAE/len(im_ids)
    RMSE = math.sqrt(RMSE/len(im_ids))
    NAE = NAE/len(im_ids)
    SRE = math.sqrt(SRE/len(im_ids))

    print("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.write("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.close()

        
