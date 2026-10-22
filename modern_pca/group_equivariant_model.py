import torch
import torch.nn as tnn

from escnn import gspaces
from escnn import nn as enn


# ============================================================
# Helpers
# ============================================================

def regular_type(gspace, n_fields: int) -> enn.FieldType:
    """
    n_fields copies of the regular representation.

    For genuine D8 (16 elements):
        1 regular field = 16 scalar channels.

    Example:
        n_fields=8 -> 128 scalar channels
    """
    return enn.FieldType(
        gspace,
        [gspace.regular_repr] * n_fields,
    )


def eq_bn(ft: enn.FieldType) -> enn.InnerBatchNorm:
    """
    Same BN convention we used in the EqNASNet benchmark.
    """
    return enn.InnerBatchNorm(
        ft,
        eps=1e-3,
        momentum=3e-4,
        affine=True,
        track_running_stats=True,
    )


# ============================================================
# Basic equivariant residual block
# ============================================================

class EqResidualBlock(tnn.Module):
    """
    x --------------------------+
    |                            |
    +-> R2Conv3x3 -> BN -> ReLU  |
        -> R2Conv3x3 -> BN -------+
                                  |
                               addition
                                  |
                                ReLU

    Input and output FieldTypes are identical.
    """

    def __init__(self, field_type: enn.FieldType):
        super().__init__()

        self.in_type = field_type
        self.out_type = field_type

        self.conv1 = enn.R2Conv(
            field_type,
            field_type,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = eq_bn(field_type)
        self.relu1 = enn.ReLU(
            field_type,
            inplace=False,
        )

        self.conv2 = enn.R2Conv(
            field_type,
            field_type,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = eq_bn(field_type)

        self.relu_out = enn.ReLU(
            field_type,
            inplace=False,
        )

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        y = y + residual
        y = self.relu_out(y)

        return y


# ============================================================
# Resolution-reducing residual block
# ============================================================

class EqDownBlock(tnn.Module):
    """
    Symmetric spatial reduction followed by residual processing.

                    +-> 1x1 projection -> BN -----+
                    |                              |
    x -> AvgPool3x3 -------------------------------+
                    |                              |
                    +-> 3x3 -> BN -> ReLU          |
                        -> 3x3 -> BN ---------------+
                                                   |
                                                addition
                                                   |
                                                 ReLU

    Pooling is done BEFORE splitting the residual paths, so both
    paths experience exactly the same spatial reduction.
    """

    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # Odd, symmetric kernel is preferable to an asymmetric
        # factorized-reduction scheme.
        self.pool = enn.PointwiseAvgPool2D(
            in_type,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Skip branch
        self.skip_proj = enn.R2Conv(
            in_type,
            out_type,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.skip_bn = eq_bn(out_type)

        # Residual branch
        self.conv1 = enn.R2Conv(
            in_type,
            out_type,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = eq_bn(out_type)
        self.relu1 = enn.ReLU(
            out_type,
            inplace=False,
        )

        self.conv2 = enn.R2Conv(
            out_type,
            out_type,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = eq_bn(out_type)

        self.relu_out = enn.ReLU(
            out_type,
            inplace=False,
        )

    def forward(self, x):
        x = self.pool(x)

        skip = self.skip_proj(x)
        skip = self.skip_bn(skip)

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        y = y + skip
        y = self.relu_out(y)

        return y


# ============================================================
# Small multiscale D8-equivariant residual network
# ============================================================

class D8MultiScaleResNetGAP(tnn.Module):
    """
    350x350 input

        stem -> 175
        pool -> F88
        down -> F44
        down -> F22
        down -> F11

    Every scale is independently converted to an invariant
    descriptor:

        feature map
            -> GroupPooling
            -> spatial global average
            -> vector

    The four scale descriptors are concatenated and classified.

    Default widths:
        F88  :  8 regular fields = 128 scalar channels
        F44  : 16 regular fields = 256 scalar channels
        F22  : 24 regular fields = 384 scalar channels
        F11  : 32 regular fields = 512 scalar channels

    The final invariant descriptor therefore has:
        8 + 16 + 24 + 32 = 80 values.
    """

    def __init__(
        self,
        num_classes=2,
        fields=(8, 16, 24, 32),
        blocks_per_scale=(1, 1, 1, 1),
        hidden_dim=None,
    ):
        super().__init__()

        assert len(fields) == 4
        assert len(blocks_per_scale) == 4
        assert all(b >= 1 for b in blocks_per_scale)

        # ----------------------------------------------------
        # 16-element dihedral symmetry:
        # 8 rotations (45 degree increments) x 2 reflection states
        # ----------------------------------------------------
        self.gspace = gspaces.flipRot2dOnR2(N=8)
        assert self.gspace.fibergroup.order() == 16
        assert self.gspace.regular_repr.size == 16

        # RGB channels are ordinary scalar fields.
        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr] * 3,
        )

        f88_type = regular_type(
            self.gspace,
            fields[0],
        )
        f44_type = regular_type(
            self.gspace,
            fields[1],
        )
        f22_type = regular_type(
            self.gspace,
            fields[2],
        )
        f11_type = regular_type(
            self.gspace,
            fields[3],
        )

        self.f88_type = f88_type
        self.f44_type = f44_type
        self.f22_type = f22_type
        self.f11_type = f11_type

        # ----------------------------------------------------
        # Stem: 350 -> 175
        # ----------------------------------------------------
        self.stem_conv = enn.R2Conv(
            self.in_type,
            f88_type,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem_bn = eq_bn(f88_type)
        self.stem_relu = enn.ReLU(
            f88_type,
            inplace=False,
        )

        # 175 -> 88
        self.stem_pool = enn.PointwiseAvgPool2D(
            f88_type,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------
        # F88 stage
        # ----------------------------------------------------
        self.stage88 = tnn.ModuleList([
            EqResidualBlock(f88_type)
            for _ in range(blocks_per_scale[0])
        ])

        # ----------------------------------------------------
        # F44 stage
        # ----------------------------------------------------
        self.down44 = EqDownBlock(
            f88_type,
            f44_type,
        )

        self.stage44 = tnn.ModuleList([
            EqResidualBlock(f44_type)
            for _ in range(blocks_per_scale[1] - 1)
        ])

        # ----------------------------------------------------
        # F22 stage
        # ----------------------------------------------------
        self.down22 = EqDownBlock(
            f44_type,
            f22_type,
        )

        self.stage22 = tnn.ModuleList([
            EqResidualBlock(f22_type)
            for _ in range(blocks_per_scale[2] - 1)
        ])

        # ----------------------------------------------------
        # F11 stage
        # ----------------------------------------------------
        self.down11 = EqDownBlock(
            f22_type,
            f11_type,
        )

        self.stage11 = tnn.ModuleList([
            EqResidualBlock(f11_type)
            for _ in range(blocks_per_scale[3] - 1)
        ])

        # ----------------------------------------------------
        # Independent group pooling at each receptive-field
        # scale.
        #
        # GroupPooling converts each regular field into one
        # trivial/invariant scalar field.
        # ----------------------------------------------------
        self.group_pool88 = enn.GroupPooling(f88_type)
        self.group_pool44 = enn.GroupPooling(f44_type)
        self.group_pool22 = enn.GroupPooling(f22_type)
        self.group_pool11 = enn.GroupPooling(f11_type)

        descriptor_dim = sum(fields)

        self.descriptor_dim = descriptor_dim
        self.num_classes = num_classes

        # ----------------------------------------------------
        # Tiny classifier
        # ----------------------------------------------------
        if hidden_dim is None:
            # Preferred first benchmark:
            # multiscale descriptor -> classes directly.
            self.classifier = tnn.Linear(
                descriptor_dim,
                num_classes,
            )
        else:
            self.classifier = tnn.Sequential(
                tnn.Linear(
                    descriptor_dim,
                    hidden_dim,
                ),
                tnn.ReLU(inplace=True),
                tnn.Linear(
                    hidden_dim,
                    num_classes,
                ),
            )

    def _wrap(self, x):
        if isinstance(x, enn.GeometricTensor):
            assert x.type == self.in_type
            return x

        assert x.ndim == 4
        assert x.shape[1] == 3

        return enn.GeometricTensor(
            x,
            self.in_type,
        )

    @staticmethod
    def _spatial_gap(x):
        """
        x has already undergone GroupPooling, so its fiber fields
        are trivial representations.

        Average over all spatial locations.

        For a square image and the discrete D8 actions used here,
        this yields a spatially invariant descriptor.
        """
        return x.tensor.mean(
            dim=(-2, -1)
        )

    def _descriptor(self, x, group_pool):
        x = group_pool(x)
        x = self._spatial_gap(x)
        return x

    def forward_features(self, x):
        x = self._wrap(x)

        # 350 -> 175
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        # 175 -> 88
        x = self.stem_pool(x)

        for block in self.stage88:
            x = block(x)

        f88 = x

        # 88 -> 44
        x = self.down44(x)

        for block in self.stage44:
            x = block(x)

        f44 = x

        # 44 -> 22
        x = self.down22(x)

        for block in self.stage22:
            x = block(x)

        f22 = x

        # 22 -> 11
        x = self.down11(x)

        for block in self.stage11:
            x = block(x)

        f11 = x

        return {
            "F88": f88,
            "F44": f44,
            "F22": f22,
            "F11": f11,
        }

    def forward(self, x, return_features=False):
        features = self.forward_features(x)

        z88 = self._descriptor(
            features["F88"],
            self.group_pool88,
        )

        z44 = self._descriptor(
            features["F44"],
            self.group_pool44,
        )

        z22 = self._descriptor(
            features["F22"],
            self.group_pool22,
        )

        z11 = self._descriptor(
            features["F11"],
            self.group_pool11,
        )

        z = torch.cat(
            [z88, z44, z22, z11],
            dim=1,
        )

        logits = self.classifier(z)

        if return_features:
            return logits, features, {
                "z88": z88,
                "z44": z44,
                "z22": z22,
                "z11": z11,
                "multiscale": z,
            }

        return logits
