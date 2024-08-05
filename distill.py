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

from models.sam_encoder3d import ImageEncoderViT3D
from models.tinyvit_encoder3d import TinyViT3D
from models.tinymoe_encoder3d import TinyViT3D as TinyMoE
from dataloader import Dataset_CSV, Union_Dataloader

# set up parser
parser = argparse.ArgumentParser()
# task
parser.add_argument('--model_name', type=str, default='tinymoe', choices = ['tinyvit','tinymoe'])
parser.add_argument('--data_name',type=str, default='totalsegmentator', choices=['totalsegmentator'])
parser.add_argument('--weights_path', type=str, default='/home/songty/working_dir/SAM-Med3D/work_dir/pretrained_weights/sam_med3d_turbo.pth')
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
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()
device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = os.path.join(args.work_dir, 'distill',args.data_name, args.model_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
LOG_OUT_DIR = os.path.join(MODEL_SAVE_PATH)

def build_model(args):
    teacher_model = ImageEncoderViT3D(
        depth=12,
        embed_dim=768,
        img_size=128,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=384,
    ).to(args.device)
    if args.model_name == 'tinyvit':
        student_model = TinyViT3D(
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
        ).to(args.device)

    elif args.model_name == 'tinymoe':
        student_model = TinyMoE(
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
            num_experts=8,
            slots_per_expert=1,
            device= args.device,
        ).to(args.device)
    else:
        raise ValueError('No model match.')
    if args.multi_gpu:
        teacher_model = DDP(teacher_model, device_ids=[args.rank], output_device=args.rank)
        student_model = DDP(student_model, device_ids=[args.rank], output_device=args.rank)
    return teacher_model,student_model

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

def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt().clone()

class BaseTrainer:
    def __init__(self, teacher_model, student_model, dataloaders, args):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.step_best_loss = np.inf
        self.losses = []
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(args.weights_path)
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_optimizer(self):
        if self.args.multi_gpu:
            model = self.student_model.module
        else:
            model = self.student_model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    
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

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            
        if last_ckpt:
            new_weights = {}
            for k,v in last_ckpt['model_state_dict'].items():
                if k.split('.')[0]=='image_encoder':
                    new_k = '.'.join(k.split('.')[1:])
                    new_weights[new_k]=v
            last_ckpt['model_state_dict'] = new_weights
            if self.args.multi_gpu:
                self.teacher_model.module.load_state_dict(last_ckpt['model_state_dict'],strict=False)
            else:
                self.teacher_model.load_state_dict(last_ckpt['model_state_dict'],strict=False)
    
            self.start_epoch = 0

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

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.teacher_model.eval()
        self.student_model.train()
        if self.args.multi_gpu:
            teacher_model = self.teacher_model.module
            student_model = self.student_model.module
        else:
            teacher_model = self.teacher_model
            student_model = self.student_model

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders,disable=True)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        for step, (image3D, _) in enumerate(tbar):
            # my_context = self.model.no_sync if self.args.multi_gpu and self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            # with my_context():

            image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
            image3D = image3D.unsqueeze(dim=1).to(device)
            with torch.no_grad():
                teacher_embedding = teacher_model(image3D)
            with amp.autocast():
                student_embedding = student_model(image3D)  # (B,384,8,8,8)
            loss = customized_mseloss(student_embedding.clone(),teacher_embedding.clone()) 
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
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
        epoch_loss /= step
        return epoch_loss
    
    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
        
            epoch_loss = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}')

                if self.args.multi_gpu:
                    state_dict = self.student_model.module.state_dict()
                else:
                    state_dict = self.student_model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(
                    epoch,
                    state_dict,
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='best'
                    )

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Total loss: {self.losses}')
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
    teacher,student = build_model(args)
    trainer = BaseTrainer(teacher,student, dataloaders, args)
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
        teacher,student = build_model(args)
        # Create trainer
        trainer = BaseTrainer(teacher,student, dataloaders, args)
        # Train
        trainer.train()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
