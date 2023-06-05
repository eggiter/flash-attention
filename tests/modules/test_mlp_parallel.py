# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/modules/test_mlp_parallel.py

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel

from flash_attn.modules.mlp import GatedMlp, ParallelGatedMlp

is_sm8x = torch.cuda.get_device_capability('cuda')[0] >= 8


# @pytest.mark.parametrize('dtype', [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
@pytest.mark.parametrize('dtype', [torch.float16])
# @pytest.mark.parametrize('world_size', [1, 2, 4, 8])
@pytest.mark.parametrize('world_size', [2])
# @pytest.mark.parametrize('sequence_parallel', [True, False])
@pytest.mark.parametrize('sequence_parallel', [False])
# @pytest.mark.parametrize('activation', [F.silu, F.sigmoid])
@pytest.mark.parametrize('activation', [F.silu])
# @pytest.mark.parametrize('dim', [1024, 4096])
@pytest.mark.parametrize('dim', [1024])
def test_mlp_parallel(dim, activation, sequence_parallel, world_size, dtype):
    # NOTE: default (rtol, atol) for dtypes:
    # bfloat16: (1.6e-2, 1e-5); fp16: (1e-3, 1e-5)
    rtol, atol = (3e-3, 3e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = f'cuda:{torch.distributed.get_rank()}'
    torch.cuda.set_device(device)
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    x_pt = torch.randn(batch_size * seqlen, dim, device=device, dtype=dtype,
                       requires_grad=True)
    # We need to generate g here so that all processes get the same gradient,
    # as rank 0 will have an extra bias that changes the RNG.
    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(x_pt) / 32
    if sequence_parallel:
        x0 = tensor_parallel.scatter_to_sequence_parallel_region(x_pt)
        x = x0.detach().clone().requires_grad_()
        x1 = x0.detach().clone().requires_grad_()
    else:
        x = x_pt.detach().clone().requires_grad_()
        x1 = x_pt.detach().clone().requires_grad_()

    model_pt = GatedMlp(dim, activation=activation, device=device, dtype=dtype, bias1=False, bias2=False)
    partition_dim = model_pt.fc1.weight.shape[0] // 2 // world_size
    model = ParallelGatedMlp(dim, parallel_state.get_tensor_model_parallel_group(),
                             activation=activation,
                             bias1=False, bias2=False,
                             sequence_parallel=sequence_parallel, device=device, dtype=dtype)
    model_ref = MlpRef(dim,
                       activation=activation,
                       bias1=False, bias2=False, device=device, dtype=dtype,
                       sequence_parallel_enabled=sequence_parallel,
                       )

    with torch.no_grad():
        model.fc1.weight.copy_(
            rearrange(rearrange(model_pt.fc1.weight, '(two o) i -> two o i', two=2)[:,
                      rank * partition_dim:(rank + 1) * partition_dim],
                      'two o i -> (two o) i')
        )
        model_ref.fc1.weight.copy_(
            rearrange(rearrange(model_pt.fc1.weight, '(two o) i -> two o i', two=2)[:,
                      rank * partition_dim:(rank + 1) * partition_dim],
                      'two o i -> (two o) i')
        )
        """
        model.fc1.bias.copy_(
            rearrange(rearrange(model_pt.fc1.bias, '(two o) -> two o', two=2)[:,
                      rank * partition_dim:(rank + 1) * partition_dim],
                      'two o -> (two o)')
        )
        model_ref.fc1.bias.copy_(
            rearrange(rearrange(model_pt.fc1.bias, '(two o) -> two o', two=2)[:,
                      rank * partition_dim:(rank + 1) * partition_dim],
                      'two o -> (two o)')
        )
        """
        model.fc2.weight.copy_(
            model_pt.fc2.weight[:, rank * partition_dim:(rank + 1) * partition_dim]
        )
        model_ref.fc2.weight.copy_(
            model_pt.fc2.weight[:, rank * partition_dim:(rank + 1) * partition_dim]
        )
        """
        if rank == 0:
            model.fc2.bias.copy_(model_pt.fc2.bias)
            model_ref.fc2.bias.copy_(model_pt.fc2.bias)
        """

    out = model(x)
    out_pt = model_pt(x_pt)
    out_ref = model_ref(x1)
    partition_batch_dim = batch_size * seqlen // world_size

    assert (out - out_ref).abs().min().item() == 0.0
    torch.testing.assert_close(
        out,
        out_ref,
        rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        out_ref,
        out_pt[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
        if sequence_parallel else out_pt,
        rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        out,
        out_pt[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
        if sequence_parallel else out_pt,
        rtol=rtol, atol=atol
    )

    out_pt.backward(g)
    out.backward(g[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
                 if sequence_parallel else g)
    out_ref.backward(g[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
                 if sequence_parallel else g)
    parallel_state.destroy_model_parallel()

    assert (x.grad - x1.grad).abs().min().item() == 0.0
    assert torch.allclose(
        x.grad,
        x1.grad,
        rtol=rtol, atol=atol
    )
    assert torch.allclose(
        x1.grad,
        x_pt.grad[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
        if sequence_parallel else x_pt.grad,
        rtol=rtol, atol=atol
    )
    assert torch.allclose(
        x.grad,
        x_pt.grad[rank * partition_batch_dim:(rank + 1) * partition_batch_dim]
        if sequence_parallel else x_pt.grad,
        rtol=rtol, atol=atol
    )

    assert (model.fc1.weight.grad - model_ref.fc1.weight.grad).abs().min().item() == 0.0
    assert torch.allclose(
        model_ref.fc1.weight.grad,
        rearrange(rearrange(model_pt.fc1.weight.grad, '(two o) i -> two o i', two=2)[:,
                  rank * partition_dim:(rank + 1) * partition_dim],
                  'two o i -> (two o) i'),
        rtol=rtol, atol=atol
    )
    assert torch.allclose(
        model.fc1.weight.grad,
        rearrange(rearrange(model_pt.fc1.weight.grad, '(two o) i -> two o i', two=2)[:,
                  rank * partition_dim:(rank + 1) * partition_dim],
                  'two o i -> (two o) i'),
        rtol=rtol, atol=atol
    )

    assert (model.fc2.weight.grad - model_ref.fc2.weight.grad).abs().min().item() == 0.0
    assert torch.allclose(
        model_ref.fc2.weight.grad,
        model_pt.fc2.weight.grad[:, rank * partition_dim:(rank + 1) * partition_dim],
        rtol=rtol, atol=atol
    )
    assert torch.allclose(
        model.fc2.weight.grad,
        model_pt.fc2.weight.grad[:, rank * partition_dim:(rank + 1) * partition_dim],
        rtol=rtol, atol=atol
    )
    return

    assert torch.allclose(
        model.fc1.bias.grad,
        rearrange(rearrange(model_pt.fc1.bias.grad, '(two o) -> two o', two=2)[:,
                  rank * partition_dim:(rank + 1) * partition_dim],
                  'two o -> (two o)'),
        rtol=rtol, atol=atol
    )
    if rank == 0:
        assert torch.allclose(model.fc2.bias.grad, model_pt.fc2.bias.grad, rtol=rtol, atol=atol)


class MlpRef(torch.nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.sigmoid,
                 bias1=True, bias2=True, multiple_of=256, return_residual=False,
                 device=None, dtype=None, sequence_parallel_enabled=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        from apex.transformer.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from apex.transformer.tensor_parallel import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(0)
        self.fc1 = ColumnParallelLinear(in_features, 2 * hidden_features, bias=bias1,
                                        gather_output=False,
                                        skip_bias_add=True,
                                        params_dtype=dtype,
                                        sequence_parallel_enabled=sequence_parallel_enabled,
                                        no_async_tensor_model_parallel_allreduce=sequence_parallel_enabled,
                                        )
        self.activation = activation
        self.fc2 = RowParallelLinear(hidden_features, out_features, bias=bias2,
                                     input_is_parallel=True,
                                     skip_bias_add=True,
                                     params_dtype=dtype,
                                     sequence_parallel_enabled=sequence_parallel_enabled,
                                     )

    def forward(self, x):
        y, _ = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y, _ = self.fc2(y)
        return y
