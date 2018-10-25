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