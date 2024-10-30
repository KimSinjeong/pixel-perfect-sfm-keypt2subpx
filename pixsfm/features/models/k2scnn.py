"""
An implementation of
    K2SNet: Learning to Make Keypoints Sub-pixel Accurate
    European Conference on Computer Vision (ECCV) 2024
Adapted from https://github.com/KimSinjeong/keypt2subpx
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# This function is borrowed from GlueFactory (https://github.com/cvg/glue-factory)
def extract_patches(
    tensor: torch.Tensor,
    required_corners: torch.Tensor,
    ps: int,
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = required_corners.long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1), corner.float()


class AttnTuner(nn.Module):
    def __init__(self, output_dim=256, use_score=True):
        super(AttnTuner, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.feat_axis = 1
        self.normalized_coordinates = False
        self.use_score = use_score

        c1, c2, self.c3 = 16, 64, output_dim
        self.conv1a = nn.Conv2d(2 if use_score else 1, c1, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(c2, self.c3, kernel_size=3, stride=1, padding=0)
        # self.logsoftargmax = SpatialSoftArgmax2d(self.normalized_coordinates)

    def forward(self, patch, scorepatch):
        B, N, C, H, W = patch.shape
        # B, N, F_ = desc.shape
        assert H == W, "Patch shape must be square"
        # P = (H // 2 +1) // 2 +1
        P = H-6

        patch = patch.view(B*N, C, H, W)
        if self.use_score:
            scorepatch = scorepatch.view(B*N, 1, H, W)
        # desc = desc.view(B*N, F_, 1, 1)

        if patch.shape[1] == 3:  # RGB
            scale = patch.new_tensor([0.299, 0.587, 0.114]).view(*([1]*self.feat_axis), 3, 1, 1)
            patch = (patch * scale).sum(self.feat_axis, keepdim=True)
        x = torch.cat([patch, scorepatch], self.feat_axis) if self.use_score else patch

        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.conv3(x)
        x = F.normalize(x, p=2, dim=self.feat_axis)

        return x.view(B, N, self.c3, P, P) # B x N x F x P x P
        # x = (x * desc).sum(dim=self.feat_axis).view(B, N, P, P) # Cosine similarity (in [-1, 1])    

        # coord = self.logsoftargmax(x) - (P-1)/2.
        # return coord # B x N x 2

class Keypt2Subpx(nn.Module):
    def __init__(self, output_dim=256, use_score=True):
        super(Keypt2Subpx, self).__init__()
        self.net = AttnTuner(output_dim, use_score)
        self.use_score = use_score
        self.patch_radius = 5

    def forward(self, keypt, img_score):
        """
            keypt : N x 2
            img_score : 3+1 x H x W
            TODO: Batch support
        """
        assert img_score.shape[0] == 4, "Image concatenated with score must have 4 channels"
        score = img_score[[3], :, :]
        img = img_score[:3, :, :]

        assert (img < (1. + 1e-5)).all() and (img > -(1. + 1e-5)).all(), "Image out of range"
        C, H, W = img.shape

        bias = torch.tensor([[self.patch_radius]*2], device=keypt.device)

        # RGB image to grayscale
        if img.shape[0] == 3: # RGB
            scale = img.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img = (img * scale).sum(0, keepdim=True)

        # Image patches
        img_padded = torch.nn.functional.pad(img, [self.patch_radius]*4, mode='constant', value=0.)
        idx = (keypt - bias + self.patch_radius).int()
        patch = extract_patches(img_padded.to(device=idx.device), idx, 2*self.patch_radius+1)[0].unsqueeze(0)

        # Score patches
        scorepatch = None
        if self.use_score:
            score_padded = torch.nn.functional.pad(score, [self.patch_radius]*4, mode='constant', value=0.)
            scorepatch = extract_patches(score_padded.to(device=idx.device), idx, 2*self.patch_radius+1)[0].unsqueeze(0)

        return self.net(patch, scorepatch).squeeze(0) # (B x -> X) N x F x P x P

class K2SCNN(BaseModel):
    default_conf = {
        'checkpointing': None,
        'output_dim': 256,
        'pretrained': 'k2scnn',
    }

    url = "https://github.com/KimSinjeong/keypt2subpx/tree/master/pretrained/k2s_spsg_pretrained.pth"

    def _init(self, conf):
        assert conf.pretrained in ['k2scnn', None]
        self.encoder = Keypt2Subpx()
        layers = self.encoder.children()

        self.output_dims = [self.conf.output_dim] # No multiscale support
        self.scales = []
        current_scale = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i == len(layers) - 1:
                self.scales.append(2**current_scale)

        if conf.pretrained == 'k2scnn':
            path = Path(__file__).parent / "checkpoints" / 'k2s_spsg_pretrained.pth'
            logger.info(f'Loading K2SCNN checkpoint at {path}.')
            if not path.exists():
                logger.info('Downloading K2SCNN weights.')
                import subprocess
                path.parent.mkdir(exist_ok=True)
                subprocess.call(["wget", self.url, "-q"],
                                cwd=path.parent)
            state_dict = torch.load(path, map_location='cpu')['state_dict']
            params = self.state_dict()  # @TODO: Check why these two lines fail
            state_dict = {k: v for k, v in state_dict.items()
                          if k in params.keys() and v.shape == params[k].shape}
            self.load_state_dict(state_dict, strict=False)

    def _forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        assert image.shape[0] == 1, "Batch size must be 1"
        assert image.shape[1] == 4, "Image concatenated with score must have 4 channels"
        return [self.encoder(image)]
