import random

import torch
import torchvision

from utilities import tn_resnet
import kornia as K


class InferenceDataAugmentation(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.transforms = torch.nn.Sequential(K.augmentation.Normalize(mean=mean, std=std))

    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)


class DataAugmentation(torch.nn.Module):
    def __init__(self, patch_size, crop_prob, hor_flip_prob, vert_flip_prob, gauss_blur_divider, cj_brightness, cj_contrast, cj_saturation, cj_hue, cj_prob, gs_prob, gauss_sigma, gauss_prob, solarize_threshold, solarize_prob, mean, std):
        super().__init__()

        # Compute the Gaussian blur kernel size.
        gb_ks = int(patch_size / gauss_blur_divider)
        gb_ks = int(gb_ks / 2)
        self.gb_ks = 2 * gb_ks + 1
        self.gauss_sigma = gauss_sigma
        self.gauss_prob = gauss_prob

        self.trans_1 = torch.nn.Sequential(K.augmentation.RandomResizedCrop(size=(patch_size, patch_size), p=crop_prob),
                                           K.augmentation.RandomHorizontalFlip(p=hor_flip_prob),
                                           K.augmentation.RandomVerticalFlip(p=vert_flip_prob),
                                           K.augmentation.ColorJitter(brightness=cj_brightness, contrast=cj_contrast, saturation=cj_saturation, hue=cj_hue, p=cj_prob),
                                           K.augmentation.RandomGrayscale(p=gs_prob))
                                              
        self.trans_2 = torch.nn.Sequential(K.augmentation.RandomSolarize(thresholds=(solarize_threshold / 255, solarize_threshold / 255), additions=(0, 0), p=solarize_prob),
                                           K.augmentation.Normalize(mean=mean, std=std))

    @torch.no_grad()
    def forward(self, x):
        x = self.trans_1(x)
        if self.gauss_prob > 0:
            sigma = random.uniform(self.gauss_sigma[0], self.gauss_sigma[1])
            x = K.augmentation.RandomGaussianBlur(sigma=(sigma, sigma), kernel_size=(self.gb_ks, self.gb_ks), p=self.gauss_prob)(x)
        return self.trans_2(x)


class LinearDataAugmentation(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self.transforms = torch.nn.Sequential(K.augmentation.RandomResizedCrop(size=(params.crop_size, params.crop_size)),
                                              K.augmentation.RandomHorizontalFlip(),
                                              K.augmentation.RandomVerticalFlip(),
                                              K.augmentation.Normalize(mean=params.mean, std=params.std))
    
    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)


class SemiSupDataAugmentation(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self.transforms = torch.nn.Sequential(K.augmentation.RandomResizedCrop(size=(params.crop_size, params.crop_size)),
                                              K.augmentation.RandomHorizontalFlip(),
                                              K.augmentation.RandomVerticalFlip(),
                                              K.augmentation.ColorJitter(brightness=params.view_1_cj_brightness, contrast=params.view_1_cj_contrast, hue=params.view_1_cj_hue, saturation=params.view_1_cj_saturation),
                                              K.augmentation.Normalize(mean=params.mean, std=params.std))
    
    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)


###############
# Model utils #
###############

def Projector(args, embedding):
    # Source: https://github.com/facebookresearch/vicreg/blob/a73f567660ae507b0667c68f685945ae6e2f62c3/main_vicreg.py#L223
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(torch.nn.Linear(f[i], f[i + 1]))
        layers.append(torch.nn.BatchNorm1d(f[i + 1]))
        layers.append(torch.nn.ReLU(True))
    layers.append(torch.nn.Linear(f[-2], f[-1], bias=False))
    return torch.nn.Sequential(*layers)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_trunc_norm=False, normalization=None):
        super().__init__()
        linear_net = tn_resnet.Linear if use_trunc_norm else torch.nn.Linear

        assert num_layers >= 0, "negative layers?!?"
        if normalization is not None:
            assert callable(normalization), "normalization must be callable"

        if num_layers == 0:
            self.net = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = linear_net(input_dim, output_dim, bias=False)
            return

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(linear_net(prev_dim, hidden_dim))
            if normalization is not None:
                layers.append(normalization())
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        layers.append(linear_net(hidden_dim, output_dim, bias=False))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_encoder(name: str, **kwargs) -> torch.nn.Module:
    """
    Gets just the encoder portion of a torchvision model (replaces final layer with identity)
    :param name: (str) name of the model
    :param kwargs: kwargs to send to the model
    :return:
    """

    if name in tn_resnet.__dict__:
        model_creator = tn_resnet.__dict__.get(name)
    elif name in torchvision.models.__dict__:
        model_creator = torchvision.models.__dict__.get(name)
    else:
        raise AttributeError(f"Unknown architecture {name}")

    assert model_creator is not None, f"no torchvision model named {name}"
    model = model_creator(**kwargs)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    else:
        raise NotImplementedError(f"Unknown class {model.__class__}")

    return model


####################
# Evaluation utils #
####################


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_softmax_with_factors(logits: torch.Tensor, log_factor: float = 1, neg_factor: float = 1) -> torch.Tensor:
    exp_sum_neg_logits = torch.exp(logits).sum(dim=-1, keepdim=True) - torch.exp(logits)
    softmax_result = logits - log_factor * torch.log(torch.exp(logits) + neg_factor * exp_sum_neg_logits)
    return softmax_result



