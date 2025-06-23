# Standard library
from enum import Enum


class CoeffGenerationMethod(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'


class GridGenerationMethod(Enum):
    MESH = 'mesh'
    UNIFORM = 'uniform'


class DeformationType(Enum):
    TRANSLATION_X = 'translation_x'
    TRANSLATION_Y = 'translation_y'
    TRANSLATION_Z = 'translation_z'
    ROTATION_X = 'rotation_x'
    ROTATION_Y = 'rotation_y'
    ROTATION_Z = 'rotation_z'


class FeaturesType(Enum):
    XYZ = 'xyz'
    RISP = 'risp'
    SHOT = 'shot'


class PoseType(Enum):
    NONE = 'none'
    PCA = 'pca'
    RANDOM_ROTATION = 'random_rotation'
    ALIGN_NORMAL_Z = 'align_normal_z'  # New option