import torch
import torch.nn as nn

class InpaintingLoss(nn.Module):
    """
    Loss function for image inpainting progress by combining:
        - Pixel loss          : L1 losses on the network output for the hole and the non-hole pixels respectively.
        
        - Perceptual loss     : L1 distances between both I_out and I_comp and the ground truth, 
                                but after projecting these images into higher level feature spaces using an ImageNet-pretrained VGG-16.
                                
        - Style loss          : Similar to the perceptual loss, but first perform an autocorrelation (Gram matrix) on each feature map before applying the L1.
        
        - Total variation loss: Smoothing penalty on R, where R is the region of 1-pixel dilation of the hole region.
    """
    
    def __init__(self, extractor, tv_loss='mean'):
        super().__init__()
        self.l1        = nn.L1Loss()
        self.extractor = extractor
        self.tv_loss   = tv_loss
 
    def forward(self, input, mask, output, gt):
        # Composite image: Raw output image but with the non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * output

        # Total variation loss
        tv_loss    = self._total_variation_loss(comp, mask)

        # Pixel loss
        hole_loss  = self.l1((1 - mask) * output, (1 - mask) * gt)
        valid_loss = self.l1(mask * output, mask * gt)

        # Extract features for perceptual and style loss
        feats_out  = self.extractor(output)
        feats_comp = self.extractor(comp)
        feats_gt   = self.extractor(gt)

        perc_loss  = sum(self.l1(o, g) + self.l1(c, g) 
                         for o, c, g in zip(feats_out, feats_comp, feats_gt))
        
        style_loss = sum(self.l1(self._gram_matrix(o), self._gram_matrix(g)) + self.l1(self._gram_matrix(c), self._gram_matrix(g)) 
                         for o, c, g in zip(feats_out, feats_comp, feats_gt))

        return {'valid': valid_loss, 'hole': hole_loss, 'perc': perc_loss, 'style': style_loss, 'tv': tv_loss}

    def _gram_matrix(self, input):
        B, C, H, W = input.size()
        features   = input.view(B, C, W * H)
        features_t = features.transpose(1, 2)
        
        # Avoid underflow for mixed precision training
        gram = torch.baddbmm(
            torch.zeros(B, C, C).type(features.type()), 
            features, features_t, 
            beta=0, alpha=1./(C * H * W), out=None
        )
        return gram

    def _dialation_holes(self, hole_mask):
        B, C, H, W    = hole_mask.shape
        dilation_conv = nn.Conv2d(C, C, 3, padding=1, bias=False).to(hole_mask)
        torch.nn.init.constant_(dilation_conv.weight, 1.0)
        
        with torch.no_grad():
            output_mask = dilation_conv(hole_mask)
        updated_holes = output_mask != 0
        return updated_holes.float()
    
    def _total_variation_loss(self, image, mask):
        hole_mask       = 1 - mask
        dilated_holes   = self._dialation_holes(hole_mask)
        
        colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
        rows_in_Pset    = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
        
        if self.tv_loss == 'sum':
            loss = torch.sum(torch.abs(colomns_in_Pset*(image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
                   torch.sum(torch.abs(rows_in_Pset   *(image[:, :, :1, :] - image[:, :, -1:, :])))
        else:
            loss = torch.mean(torch.abs(colomns_in_Pset*(image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
                   torch.mean(torch.abs(rows_in_Pset   *(image[:, :, :1, :] - image[:, :, -1:, :])))
        return loss