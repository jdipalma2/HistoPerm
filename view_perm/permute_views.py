import math

import torch


class PermuteViews:
    def __init__(self, shuffle_percentage, num_classes, view_1_data_transform, view_2_data_transform):
        self.shuffle_percentage = shuffle_percentage
        self.num_classes = num_classes
        self.view_1_data_transform = view_1_data_transform
        self.view_2_data_transform = view_2_data_transform

    @torch.no_grad()
    def permute_batch(self, x, y):
        x_1 = self.view_1_data_transform(x.detach().clone())
        x_2 = self.view_2_data_transform(x.detach().clone())

        sp = math.ceil(self.shuffle_percentage * len(x))

        # Only worry about a subset of the list for shuffling.
        cl = y[:sp].cpu()

        shuf = torch.zeros(self.num_classes, len(cl), dtype=torch.int64, requires_grad=False)
        # Shuffle the images based on classes.
        for c in range(self.num_classes):
            class_c = (cl == c).nonzero(as_tuple=True)[0]

            shuf[c][class_c] = class_c[torch.randperm(class_c.numel())]

        shuf = shuf.to(device=x.device, non_blocking=True)
        # Shuffle one of the views.
        # Doesn't matter which one.
        # Arbitrary choice to shuffle the first set of views.
        x_1[:sp] = x_1[:sp][torch.max(shuf, dim=0)[0]]

        # Now shuffle everything to avoid learning the first part is weird
        # in terms of views not coming from the same source.
        rand_shuf = torch.randperm(n=len(x), device=x.device)
        x_1 = x_1[rand_shuf]
        x_2 = x_2[rand_shuf]
        return x_1, x_2
