import numpy as np
import random
import datetime
import logging
import os
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from functools import partial
import transformers
import torch.nn.functional as F
from monai.losses import DiceCELoss
import matplotlib.pyplot as plt

from models.sam_encoder3d import ImageEncoderViT3D
from models.tinyvit_encoder3d import TinyViT3D
from models.tinymoe_encoder3d import TinyViT3D as TinyMoE
from dataloader import Dataset_CSV, Union_Dataloader
from segment_anything.click_method import get_next_click3D_torch_2
from segment_anything.build_sam3D import build_sam3D_vit_b_ori

# set up parser
parser = argparse.ArgumentParser()
# task
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_name', type=str, default='tinyvit', choices = ['tinyvit','tinymoe'])
parser.add_argument('--data_name',type=str, default='totalsegmentator', choices=['totalsegmentator'])
parser.add_argument('--encoder_ckpt_path', type=str,default='/home/songty/working_dir/MyProject/work_dir/distill/totalsegmentator/tinyvit/best_20240131.pth')
parser.add_argument('--model_ckpt_path', type=str,default='/home/songty/working_dir/SAM-Med3D/work_dir/pretrained_weights/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# train
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[30, 60])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--num_expert', type=int, default=64)
parser.add_argument('--slots_per_expert', type=int, default=1)

args = parser.parse_args()
device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = os.path.join(args.work_dir, 'finetune',args.data_name, args.model_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
LOG_OUT_DIR = os.path.join(MODEL_SAVE_PATH)

click_methods = {
    'random': get_next_click3D_torch_2,
}

def build_model(args):
    if args.model_name == 'tinyvit':
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
    elif args.model_name == 'tinymoe':
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
    else:
        raise ValueError('No model match.')
    model = build_sam3D_vit_b_ori(encoder=encoder).to(args.device)
    if args.multi_gpu:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    return model

def get_dataloaders(args):
    if args.data_name == 'totalsegmentator':
        train_dataset = Dataset_CSV(
            csv_folder=os.path.join('/home/songty/working_dir/SAM-Med3D/data',args.data_name),
            transform=tio.Compose([
                tio.ToCanonical(),
                tio.CropOrPad(mask_name='label', target_shape=(128,128,128)), # crop only object region
                tio.RandomFlip(axes=(0, 1, 2)),
            ]),
            threshold=1000,
            mode='train',
            expert='all',
        )
    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader

class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(args.encoder_ckpt_path,args.model_ckpt_path)
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model
        self.optimizer = torch.optim.AdamW([
            {'params': model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer,
                                                                             2,
                                                                             self.args.num_epochs)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, encoder_ckpt_path,model_ckpt_path):
        if self.args.multi_gpu:
            dist.barrier()
            encoder_ckpt = torch.load(encoder_ckpt_path, map_location=args.device)
            model_ckpt = torch.load(model_ckpt_path, map_location=args.device)
        else:
            encoder_ckpt = torch.load(encoder_ckpt_path, map_location=args.device)
            model_ckpt = torch.load(model_ckpt_path, map_location=args.device)

        new_weights = {}
        for k,v in encoder_ckpt['model_state_dict'].items():
            new_k = 'image_encoder.'+k
            new_weights[new_k]=v
        filter_weights = {}
        for k,v in model_ckpt['model_state_dict'].items():
            if k.split('.')[0]!='image_encoder':
                filter_weights[k] = v
        filter_weights.update(new_weights)
        if self.args.multi_gpu:
            self.model.module.load_state_dict(filter_weights)  
        else:
            self.model.load_state_dict(filter_weights)  
        
        self.start_epoch = 0
        print(f"Start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "best_loss": self.best_loss,
            "args": self.args,
            "used_datas": self.args.data_name,
        }, os.path.join(MODEL_SAVE_PATH, f"{describe}_{self.args.cur_day}.pth"))

    def batch_forward(self, model, image_embedding, gt3D, low_res_masks, points=None):
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)
        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)
        self.click_points.append(points_co)
        self.click_labels.append(points_la)
        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device) # (B,1,128,128,128)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D) # point_input:(B,3), label_imput(B,1)
            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()
    
    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders,disable=True)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        for step, (image3D, gt3D) in enumerate(tbar):
            my_context = self.model.no_sync if self.args.multi_gpu and self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1).to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)  # (B,384,8,8,8)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)
                epoch_loss += loss.item()

                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()
                
            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
        epoch_loss /= step

        return epoch_loss, epoch_iou, epoch_dice, pred_list
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()
    

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='best'
                    )

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {self.args.data_name}')
        logger.info('=====================================================================')

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device("cuda")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    cur_day = ''.join(cur_time.split('-')[:3])
    args.cur_day = cur_day
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()

def main():
    
    device_config(args)
    if args.multi_gpu:
        mp.set_sharing_strategy('file_system')
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        cur_day = ''.join(cur_time.split('-')[:3])
        args.cur_day = cur_day
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
