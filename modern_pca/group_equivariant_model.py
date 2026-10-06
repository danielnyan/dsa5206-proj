"""Model definitions copied from inbox/datasci-proj/group-equivariant-model.ipynb.

Only notebook execution, comments and docstrings removed; architecture unchanged.
Notebook SHA-256: f5531e70d2ef4bc237faebf31c63326d6775d194913e990cbc95b066e67eb891
"""
import torch
import torch.nn as tnn
from escnn import gspaces
from escnn import nn as enn

def regular_type(gspace, n_fields: int) -> enn.FieldType:
    return enn.FieldType(gspace, [gspace.regular_repr] * n_fields)

def eq_bn(ft: enn.FieldType) -> enn.InnerBatchNorm:
    return enn.InnerBatchNorm(ft, eps=0.001, momentum=0.0003, affine=True, track_running_stats=True)

class EqResidualBlock(tnn.Module):

    def __init__(self, field_type: enn.FieldType):
        super().__init__()
        self.in_type = field_type
        self.out_type = field_type
        self.conv1 = enn.R2Conv(field_type, field_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = eq_bn(field_type)
        self.relu1 = enn.ReLU(field_type, inplace=False)
        self.conv2 = enn.R2Conv(field_type, field_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = eq_bn(field_type)
        self.relu_out = enn.ReLU(field_type, inplace=False)

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

class EqDownBlock(tnn.Module):

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.pool = enn.PointwiseAvgPool2D(in_type, kernel_size=3, stride=2, padding=1)
        self.skip_proj = enn.R2Conv(in_type, out_type, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip_bn = eq_bn(out_type)
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = eq_bn(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=False)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = eq_bn(out_type)
        self.relu_out = enn.ReLU(out_type, inplace=False)

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

class D8MultiScaleResNetGAP(tnn.Module):

    def __init__(self, num_classes=2, fields=(8, 16, 24, 32), blocks_per_scale=(1, 1, 1, 1), hidden_dim=None):
        super().__init__()
        assert len(fields) == 4
        assert len(blocks_per_scale) == 4
        assert all((b >= 1 for b in blocks_per_scale))
        self.gspace = gspaces.flipRot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        f88_type = regular_type(self.gspace, fields[0])
        f44_type = regular_type(self.gspace, fields[1])
        f22_type = regular_type(self.gspace, fields[2])
        f11_type = regular_type(self.gspace, fields[3])
        self.f88_type = f88_type
        self.f44_type = f44_type
        self.f22_type = f22_type
        self.f11_type = f11_type
        self.stem_conv = enn.R2Conv(self.in_type, f88_type, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = eq_bn(f88_type)
        self.stem_relu = enn.ReLU(f88_type, inplace=False)
        self.stem_pool = enn.PointwiseAvgPool2D(f88_type, kernel_size=3, stride=2, padding=1)
        self.stage88 = tnn.ModuleList([EqResidualBlock(f88_type) for _ in range(blocks_per_scale[0])])
        self.down44 = EqDownBlock(f88_type, f44_type)
        self.stage44 = tnn.ModuleList([EqResidualBlock(f44_type) for _ in range(blocks_per_scale[1] - 1)])
        self.down22 = EqDownBlock(f44_type, f22_type)
        self.stage22 = tnn.ModuleList([EqResidualBlock(f22_type) for _ in range(blocks_per_scale[2] - 1)])
        self.down11 = EqDownBlock(f22_type, f11_type)
        self.stage11 = tnn.ModuleList([EqResidualBlock(f11_type) for _ in range(blocks_per_scale[3] - 1)])
        self.group_pool88 = enn.GroupPooling(f88_type)
        self.group_pool44 = enn.GroupPooling(f44_type)
        self.group_pool22 = enn.GroupPooling(f22_type)
        self.group_pool11 = enn.GroupPooling(f11_type)
        descriptor_dim = sum(fields)
        self.descriptor_dim = descriptor_dim
        self.num_classes = num_classes
        if hidden_dim is None:
            self.classifier = tnn.Linear(descriptor_dim, num_classes)
        else:
            self.classifier = tnn.Sequential(tnn.Linear(descriptor_dim, hidden_dim), tnn.ReLU(inplace=True), tnn.Linear(hidden_dim, num_classes))

    def _wrap(self, x):
        if isinstance(x, enn.GeometricTensor):
            assert x.type == self.in_type
            return x
        assert x.ndim == 4
        assert x.shape[1] == 3
        return enn.GeometricTensor(x, self.in_type)

    @staticmethod
    def _spatial_gap(x):
        return x.tensor.mean(dim=(-2, -1))

    def _descriptor(self, x, group_pool):
        x = group_pool(x)
        x = self._spatial_gap(x)
        return x

    def forward_features(self, x):
        x = self._wrap(x)
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)
        for block in self.stage88:
            x = block(x)
        f88 = x
        x = self.down44(x)
        for block in self.stage44:
            x = block(x)
        f44 = x
        x = self.down22(x)
        for block in self.stage22:
            x = block(x)
        f22 = x
        x = self.down11(x)
        for block in self.stage11:
            x = block(x)
        f11 = x
        return {'F88': f88, 'F44': f44, 'F22': f22, 'F11': f11}

    def forward(self, x, return_features=False):
        features = self.forward_features(x)
        z88 = self._descriptor(features['F88'], self.group_pool88)
        z44 = self._descriptor(features['F44'], self.group_pool44)
        z22 = self._descriptor(features['F22'], self.group_pool22)
        z11 = self._descriptor(features['F11'], self.group_pool11)
        z = torch.cat([z88, z44, z22, z11], dim=1)
        logits = self.classifier(z)
        if return_features:
            return (logits, features, {'z88': z88, 'z44': z44, 'z22': z22, 'z11': z11, 'multiscale': z})
        return logits
