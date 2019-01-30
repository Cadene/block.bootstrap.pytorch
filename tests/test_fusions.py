import pytest
import torch
import inspect
import itertools
from block.models.networks.fusions import fusions

bsize = 2
x_arg = [
    [torch.randn(bsize, 10), torch.randn(bsize, 20)],
    [torch.randn(bsize, 20), torch.randn(bsize, 10)],
    [torch.randn(bsize, 10), torch.randn(bsize, 10)]
]
F_arg = [F for k, F in fusions.__dict__.items() \
         if inspect.isclass(F) and k != 'ConcatMLP' and k != 'MCB']
args = [(F, x) for F, x in itertools.product(F_arg, x_arg)]

@pytest.mark.parametrize('F, x', args)
def test_fusions(F, x):
    input_dims = [x[0].shape[-1], x[1].shape[-1]]
    output_dim = 2
    fusion = F(
        input_dims,
        output_dim,
        mm_dim=20)
    out = fusion(x)
    if torch.cuda.is_available():
        fusion.cuda()
        out = fusion([x[0].cuda(), x[1].cuda()])
    assert torch.Size([2,2]) == out.shape

@pytest.mark.parametrize('x', x_arg)
def test_ConcatMLP(x):
    input_dims = [x[0].shape[-1], x[1].shape[-1]]
    output_dim = 2
    fusion = fusions.ConcatMLP(
        input_dims,
        output_dim,
        dimensions=[5,5])
    out = fusion(x)

@pytest.mark.mcb
@pytest.mark.parametrize('x', x_arg)
def test_MCB(x):
    input_dims = [x[0].shape[-1], x[1].shape[-1]]
    output_dim = 2
    fusion = fusions.MCB(
        input_dims,
        output_dim,
        mm_dim=100)
    if torch.cuda.is_available():
        fusion.cuda()
        out = fusion([x[0].cuda(), x[1].cuda()])
