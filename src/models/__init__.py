"""
Model registry for the deep change-detection models (Section 2). train.py and
predict.py import build_model / is_fixed_n from here, so adding a model in Task 6
or Task 8 is a one-line change in ONE place. This is the "dispatch on a model name
-> builder" pattern set by this first deep model (Section 7: the architecture is
the only variable, the training recipe is identical for every model).

Fixed-N models take padded [B, N, 3] cube input plus a real-point mask
(siamese_pointnet here; siamese_pointnet2 and siamgcn join in Task 6, all via
collate_cubes_fixed_n). Variable-N models (siamese_kpconv grid, randla large-N)
take ragged native-length clouds and also arrive in Task 6; they are intentionally
NOT registered yet so the fixed-N path stays clean.
"""

from .siamese_pointnet import SiamesePointNet
from .siamese_pointnet2 import SiamesePointNet2
from .siamgcn import SiamGCN
from .siamese_kpconv import SiameseKPConv
from .randla import RandLA

# Models that consume fixed-N padded input + mask (the collate_cubes_fixed_n path).
# siamese_kpconv and randla are intentionally NOT here: both are variable-N (ragged
# native clouds, per-cube forward), so is_fixed_n stays False for them and the
# loaders/train/predict route them through the variable-N path.
FIXED_N_MODELS = {"siamese_pointnet", "siamese_pointnet2", "siamgcn"}


def build_model(model_name):
    """Return a freshly constructed model for the given Section 2 model name."""
    if model_name == "siamese_pointnet":
        return SiamesePointNet()
    if model_name == "siamese_pointnet2":
        return SiamesePointNet2()
    if model_name == "siamgcn":
        return SiamGCN()
    if model_name == "siamese_kpconv":
        return SiameseKPConv()
    if model_name == "randla":
        # RandLA-Net (Task 8): variable-N, MS/HKCD only (excluded from LD for density).
        return RandLA()
    raise ValueError("unknown or not-yet-implemented model: " + model_name)


def is_fixed_n(model_name):
    """True if the model uses the fixed-N padded dataloader/collate path."""
    return model_name in FIXED_N_MODELS
