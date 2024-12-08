import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConvolution(nn.Conv2d):
    """
    Partial Convolution for Image Inpainting, where: 
        - Convolution operation: Conditioned only on valid pixels.
        - Mask update          : Mask is updated and normalized after each operation.
        
    => Output values depend only on the unmasked inputs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Kernel for mask update, same size as conv filter
        self.register_buffer(
            'mask_kernel', 
            torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        )

        # Calculate sum(1), used for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] * self.mask_kernel.shape[3]
        
        # Initialize conv weights for better convergence
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):      
        with torch.no_grad():
            # Update mask 
            update_mask = F.conv2d(mask, self.mask_kernel, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

            # Calculate mask ratio to normalize convolution outputs
            mask_ratio  = self.sum1 / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio  = mask_ratio * update_mask

        # Perform pconv on the masked image
        # Calculate W.T (.) (X * M)
        conved = F.conv2d(img * mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Renormalize output based on valid mask regions
        if self.bias is not None:
            # Calculate W.T (.) (X * M) + sum(1) / sum(M) + bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output    = mask_ratio * (conved - bias_view) + bias_view
            
            output    = output * update_mask # chưa rõ là * update_mask hay mask_ratio thì đúng hơn
        else:
            output = conved * mask_ratio

        return output, update_mask

class UpsampleData(nn.Module):
    """
    Nearest-neighbor upsampling layer with concatenation of encoder and decoder features/masks.
    """
    
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, dec_feature, enc_feature, dec_mask, enc_mask):
        out      = torch.cat([self.upsample(dec_feature), enc_feature], dim=1)
        out_mask = torch.cat([self.upsample(dec_mask)   , enc_mask]   , dim=1)
        return out, out_mask

class PConvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, sample='none-3', dec=False, bn=True, active='relu', conv_bias=False):
        super().__init__()

        kernel_size, stride, padding = {
            'down-7': (7, 2, 3), 
            'down-5': (5, 2, 2), 
            'down-3': (3, 2, 1)
        }.get(sample, (3, 1, 1))
        self.conv = PartialConvolution(in_ch, out_ch, kernel_size, stride, padding, bias=conv_bias)

        # Perform upsamping if in the decoding stage
        if dec:
            self.upcat = UpsampleData()
            
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)

        if active == 'relu':
            self.activation = nn.ReLU()
        elif active == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, img, mask, enc_img=None, enc_mask=None):
        # Perform upsampling and concatenation if in decoding stage
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(img, enc_img, mask, enc_mask)
            out, update_mask = self.conv(out, update_mask)
        else:
            out, update_mask = self.conv(img, mask)

        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
            
        return out, update_mask