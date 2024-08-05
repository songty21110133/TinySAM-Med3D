from glob import glob
import argparse
import os
import torchio as tio
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import pickle
import json

from dataloader import Dataset_CSV, Union_Dataloader
from segment_anything.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything.build_sam3d import build_sam3D_vit_b_ori
from models.tinyvit_encoder3d import TinyViT3D
from models.tinymoe_encoder3d import TinyViT3D as TinyMoE
# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--data_name',type=str, default='totalsegmentator', choices=['totalsegmentator'])
parser.add_argument('--data_expert', type=str,default='liver')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--encoder_name', type=str,default='tinymoe', choices=['tinyvit','tinymoe'])
parser.add_argument('--encoder_ckpt_path', type=str)
parser.add_argument('--model_ckpt_path', type=str,default='/home/songty/working_dir/SAM-MoE/work_dir/pretrained_weights/sam_med3d_turbo.pth')
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--load_encoder', action='store_true', default=False)
parser.add_argument('--num_expert', type=int, default=8)
parser.add_argument('--slots_per_expert', type=int, default=1)
# task
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

def build_model(args):
    if args.encoder_name == 'tinyvit':
        encoder = TinyViT3D(
            img_size=128, 
            in_chans=1, 
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
    elif args.encoder_name == 'tinymoe':
        encoder = TinyMoE(
            img_size=128, 
            in_chans=1, 
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
            num_experts=args.num_expert,
            slots_per_expert=args.slots_per_expert,
            device= args.device,
        )
    model = build_sam3D_vit_b_ori(encoder=encoder).to(args.device)
    return model

# set up parser

def load_checkpoint(model,args):
    if args.load_encoder:
        encoder_ckpt = torch.load(args.encoder_ckpt_path, map_location=args.device)
        model_ckpt = torch.load(args.model_ckpt_path, map_location=args.device)
        new_weights = {}
        for k,v in encoder_ckpt['model_state_dict'].items():
            new_k = 'image_encoder.'+k
            new_weights[new_k]=v
        filter_weights = {}
        for k,v in model_ckpt['model_state_dict'].items():
            if k.split('.')[0]!='image_encoder':
                filter_weights[k] = v
        filter_weights.update(new_weights)
        model.load_state_dict(filter_weights)
    else:
        model_ckpt = torch.load(args.model_ckpt_path, map_location=args.device)
        model.load_state_dict(model_ckpt['model_state_dict'])
    return model

def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def device_config(args):
    try:
        if args.use_gpu:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
    except RuntimeError as e:
        print(e)

def finetune_model_predict3D(img3D, gt3D, sam_model_tune, norm_transform, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            if(num_click>1):
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    return pred_list, click_points, click_labels, iou_list, dice_list

@torch.no_grad()
def test(model,dataloader):
    model.eval()
    all_iou_list = []
    all_dice_list = []
    out_dice = dict()
    out_dice_all = OrderedDict()
    for batch_data in tqdm(dataloader):
        image3D, gt3D, img_name = batch_data
        sz = image3D.size()
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)

        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        seg_mask_list, points, labels, iou_list, dice_list = finetune_model_predict3D(
            image3D, gt3D, model, norm_transform, device=args.device,
            click_method=args.point_method, num_clicks=args.num_clicks,
            prev_masks=None)
        
        per_iou = max(iou_list)
        all_iou_list.append(per_iou)
        all_dice_list.append(max(dice_list))
        print(dice_list)
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f'{i}'] = dice
        out_dice_all[img_name[0]] = cur_dice_dict

    return all_iou_list,all_dice_list,out_dice,out_dice_all




def main():
    device_config(args)
    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size,args.crop_size,args.crop_size)),
    ]

    test_dataset = Dataset_CSV(
        csv_folder=os.path.join('/home/songty/working_dir/SAM-MoE/data',args.data_name),
        mode="test",
        transform=tio.Compose(infer_transform),
        threshold=0,
        pcc=False,
        expert=args.data_expert,
    )

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=False
    )
    model = build_model(args)
    load_checkpoint(model,args)


    all_iou_list,all_dice_list,out_dice,out_dice_all = test(model,test_dataloader)

    print('Mean IoU : ', sum(all_iou_list)/len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list)/len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if(args.split_num>1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print(f"{args.data_expert} Done")

if __name__ == "__main__":
    main()



