def load_ckpt(ckpt_path, models, optimizers=None, for_predict=False):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])

    if for_predict: 
        model.eval()
        return model
    else:
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('step', 0)