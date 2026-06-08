import torch
import segmentation_models_pytorch as smp


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_model(
    encoder_name="resnet34",
    encoder_weights="imagenet",#"imagenet", #None for no pretraining, "imagenet" for ImageNet pretraining, or a path to a custom checkpoint
    in_channels=1,
    classes=1,
):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        decoder_use_norm="batchnorm",
    )
    return model


def load_model(model_path, device=None):
    if device is None:
        device = get_device()

    model = create_model()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


