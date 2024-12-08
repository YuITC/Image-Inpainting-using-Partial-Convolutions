import torch
import torch.nn as nn

from src.models.PConvLayer import PConvLayer

class PConvUNet(nn.Module):
    """
    Partial Convolution with U-Net structure.
    Combines encoder and decoder with skip connections to handle irregular masks.
    """
    
    def __init__(self, finetune=False, in_ch=3, layer_size=6):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False # Freeze BatchNorm in fine-tuning
        self.layer_size    = layer_size                  # Number of layers in encoder/decoder

        # Encoder layers
        self.enc_1 = PConvLayer(in_ch, 64, 'down-7', bn=False)
        self.enc_2 = PConvLayer(64,   128, 'down-5')
        self.enc_3 = PConvLayer(128,  256, 'down-5')
        self.enc_4 = PConvLayer(256,  512, 'down-3')
        self.enc_5 = PConvLayer(512,  512, 'down-3')
        self.enc_6 = PConvLayer(512,  512, 'down-3')
        self.enc_7 = PConvLayer(512,  512, 'down-3')
        self.enc_8 = PConvLayer(512,  512, 'down-3')

        # Decoder layers
        self.dec_8 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_7 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_6 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_5 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_4 = PConvLayer(512 + 256, 256, dec=True, active='leaky')
        self.dec_3 = PConvLayer(256 + 128, 128, dec=True, active='leaky')
        self.dec_2 = PConvLayer(128 + 64,   64, dec=True, active='leaky')
        self.dec_1 = PConvLayer(64  + 3,     3, dec=True, bn=False, active=None, conv_bias=True)

    def forward(self, img, mask):
        # Store intermediate encoder features and masks
        enc_f, enc_m = [img], [mask]

        # Encoder forward pass
        for layer_num in range(1, self.layer_size + 1):
            if layer_num == 1:
                feature, update_mask = getattr(self, f'enc_{layer_num}')(img, mask)
            else:
                enc_f.append(feature)
                enc_m.append(update_mask)
                feature, update_mask = getattr(self, f'enc_{layer_num}')(feature, update_mask)

        # Decoder forward pass
        for layer_num in reversed(range(1, self.layer_size + 1)):
            feature, update_mask = getattr(self, f'dec_{layer_num}')(feature, update_mask, enc_f.pop(), enc_m.pop())

        return feature, update_mask

    def train(self, mode=True):
        super().train(mode)
        if not self.freeze_enc_bn:
            return 
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()
                
# # Testing
# batch_size    = 2
# in_channels   = 3
# height, width = 256, 256

# img  = torch.rand(batch_size, in_channels, height, width)
# mask = torch.ones(batch_size, in_channels, height, width)

# model = PConvUNet(in_ch=in_channels, layer_size=6)
# if torch.cuda.is_available():
#     model = model.cuda()
#     img   = img.cuda()
#     mask  = mask.cuda()

# output, updated_mask = model(img, mask)

# assert output.shape == img.shape, f"Expected output shape {img.shape}, got {output.shape}"
# assert updated_mask.shape == mask.shape, f"Expected updated_mask shape {mask.shape}, got {updated_mask.shape}"
# print("Test passed: PConvUNet forward pass is working correctly.")