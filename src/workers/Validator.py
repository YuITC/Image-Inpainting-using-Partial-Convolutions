import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.utils import save_image, make_grid
from torch.amp import autocast
from torch.nn import functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from torchmetrics.functional import peak_signal_noise_ratio as psnr
from tqdm import tqdm

class Validator:
    def __init__(self, config, device, model, dataset_val, criterion, optimizer, writer):
        self.config         = config
        self.device         = device
        self.model          = model
        self.dataset_val    = dataset_val
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.writer         = writer
        self.best_loss      = float('inf')
        # self.best_psnr_ssim = 0.0
        
    def create_sub_dataset_val(self, pct=0.1):
        dataset_size = len(self.dataset_val.dataset)
        num_samples  = int(pct * dataset_size)
        subset       = Subset(self.dataset_val.dataset, random.sample(range(dataset_size), num_samples))
        
        return DataLoader(subset, batch_size=self.config.batch_size, shuffle=False)

    def evaluate_loss(self, dataset_val, step):
        visualize_flag = False
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataset_val, desc="Validation Progress (by loss)", unit="batch", position=0, leave=True):
                input, mask, gt = [x.to(self.device, non_blocking=True) for x in batch]
                with autocast(device_type='cuda'):
                    output, _ = self.model(input, mask)
                    
                if not visualize_flag:
                    visualize_flag = True
                    comp = mask * input + (1 - mask) * output
                    save_image(
                        make_grid(torch.cat([input.cpu(), mask.cpu(), output.cpu(), comp.cpu(), gt.cpu()], dim=0), 
                                  nrow=self.config.batch_size, padding=2), 
                        f'{self.config.ckpt}/val_vis/{step}.png'
                    )
                    
                loss_dict   = self.criterion(input, mask, output, gt)
                total_loss += sum(getattr(self.config, f"{key}_coef") * val for key, val in loss_dict.items()).item()
        
        return total_loss / len(dataset_val)

    # def evaluate_psnr_ssim(self, dataset_val):
    #     pass

    def save_best_model(self, step, criterion, value):
        print(f"> New best model by {criterion} saved with evaluate score {value:.6f}")
        torch.save({
            'step'             : step,
            'model'            : self.model.state_dict(),
            'optimizer'        : self.optimizer.state_dict(),
            f'best_{criterion}': value,
        }, f"{self.config.ckpt}/models/best_model_by_{criterion}.pth")

    def evaluate(self, step):
        print(f"- Evaluation at [STEP: {step}] ...")
        self.model.eval()

        sub_val = self.create_sub_dataset_val(pct=0.04)

        # Evaluate loss
        avg_loss = self.evaluate_loss(sub_val, step)
        print(f"> Validation Loss: {avg_loss:.6f}")
        self.writer.add_scalar("Validation/Avg_Loss", avg_loss, step)
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_best_model(step, "loss", avg_loss)

        # Evaluate PSNR and SSIM
        # avg_psnr_ssim = self.evaluate_psnr_ssim(sub_val)
        # self.writer.add_scalar("Validation/PSNR & SSIM", avg_psnr_ssim, step)
        # if avg_psnr_ssim > self.best_psnr_ssim:
        #     self.best_psnr_ssim = avg_psnr_ssim
        #     self.save_best_model(step, "psnr_ssim", avg_psnr_ssim)