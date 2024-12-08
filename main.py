# Thư viện tiêu chuẩn
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision import transforms

from src.config.Config                import Config
from src.datasets.MaskGenerator       import MaskGenerator
from src.datasets.InitDataset         import InitDataset
from src.models.InpaintingLoss        import InpaintingLoss
from src.models.VGG16FeatureExtractor import VGG16FeatureExtractor
from src.models.PConvUNet             import PConvUNet
from src.workers.Trainer              import Trainer
# from src.workers.Tester               import Tester

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
set_seed(42)

# For faster training with fixed-size inputs
torch.backends.cudnn.benchmark = True  

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # Configuring
    print("Configuring ...")
    config      = Config('src/config/batch4_iter50k_save5k.yml')
    config.ckpt = "outputs"
    os.makedirs(os.path.join(config.ckpt, "logs"), exist_ok=True)
    os.makedirs(os.path.join(config.ckpt, "models") , exist_ok=True)

    # Setup device
    print("Setup device ...")
    device = torch.device(f"cuda:{config.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = PConvUNet(finetune=config.finetune, layer_size=config.layer_size)
    if config.finetune:
        model.load_state_dict(torch.load(config.finetune)['model'])
    model.to(device)

    img_train_tf  = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                        transforms.ToTensor()])
    img_val_tf  = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    mask_tf     = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    maskGenerator = MaskGenerator(256, 256)

    # Load Trainer
    trainer = Trainer(
        step   = 0,
        config = config,
        device = device,
        model  = model,
        
        dataset_train = InitDataset(config.data_root, img_train_tf, mask_tf, data="train", maskGenerator=maskGenerator),
        dataset_val   = InitDataset(config.data_root, img_val_tf  , mask_tf, data="val"  , maskGenerator=maskGenerator),
        
        criterion = InpaintingLoss(VGG16FeatureExtractor(), tv_loss=config.tv_loss).to(device),
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = config.finetune_lr if config.finetune else config.initial_lr,
            weight_decay = config.weight_decay
        )
    )
    trainer.iterate()