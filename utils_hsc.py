import torch

def save_model(path, encoder, decoder, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


def save_xai_module(path, CaptionEnc, Guide, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'CaptionEnc_state': CaptionEnc.state_dict(),
            'Guide_state': Guide.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer1_state'] = optimizer[0].state_dict()
        model_dict['optimizer2_state'] = optimizer[1].state_dict()

    torch.save(model_dict, path)