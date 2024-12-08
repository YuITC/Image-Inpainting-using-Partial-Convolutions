import os
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.workers.Validator import Validator

class Trainer:
    def __init__(self, step, config, device, model, dataset_train, dataset_val, criterion, optimizer):
        self.stepped       = step
        self.config        = config
        self.device        = device
        self.model         = model.to(device)
        self.dataset_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True , num_workers=os.cpu_count(), pin_memory=True)
        self.dataset_val   = DataLoader(dataset_val  , batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        self.criterion     = criterion.to(device)
        self.optimizer     = optimizer
        self.writer        = SummaryWriter(log_dir=config.ckpt)
        self.scaler        = GradScaler('cuda')
        self.validator     = Validator(model=self.model, dataset_val=self.dataset_val, criterion=self.criterion, 
                                       device=self.device, config=self.config, optimizer=self.optimizer, writer=self.writer)

    def iterate(self):
        # torch.autograd.set_detect_anomaly(True)
        
        progress_bar = tqdm(total=self.config.max_iter, desc="Training Progress", unit="step", position=0, leave=True)
        for step, (input, mask, gt) in enumerate(self.dataset_train):
            current_step = step + self.stepped
            loss_dict    = self.train(current_step, input, mask, gt)

            if current_step % self.config.log_interval == 0:
                self.report(current_step, loss_dict)
                
            if current_step % self.config.vis_interval == 0:
                self.validator.evaluate(current_step)
                
            if current_step % self.config.save_model_interval == 0:
                self.checkpoint(current_step)
                
            if current_step >= self.config.max_iter:
                print("- Max iterations reached. Stop training!")
                break
                
            progress_bar.update(1)
        progress_bar.close()

    def train(self, step, input, mask, gt):
        self.model.train()
        input, mask, gt = [x.to(self.device, non_blocking=True) for x in [input, mask, gt]]

        self.optimizer.zero_grad()
        with autocast(device_type='cuda'):
            output, _  = self.model(input, mask)
            loss_dict  = self.criterion(input, mask, output, gt)
            total_loss = sum(getattr(self.config, f"{key}_coef") * val for key, val in loss_dict.items())

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        loss_dict['total'] = total_loss
        return {k: v.item() for k, v in loss_dict.items()}
        
    def report(self, step, loss_dict):
        print(f"[STEP: {step}]\t Total Loss: {loss_dict['total']:.6f}")
        
        for key, val in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', val, step)

    def checkpoint(self, step):
        print(f"- Checkpoint at [STEP: {step}]")

        torch.save({
            'step'     : step,
            'model'    : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{self.config.ckpt}/models/{step}.pth")