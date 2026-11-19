"""Optional reproducible evaluation contexts for caller-owned model state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from shapiq_games._optional import require

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager


def torch_cpu_context(model: object) -> Callable[[int], AbstractContextManager[None]]:
    """Build a seeded evaluation context for a CPU Torch model.

    Every entry snapshots all module training flags, puts the model in evaluation mode, and uses
    no-grad with a CPU RNG seeded from a separate CPU ``Generator``. Exit restores the complete
    CPU RNG state and each module's original training flag, including mixed training/evaluation
    configurations and failures. Accelerator RNG state is neither seeded nor initialized.

    This is a bounded CPU evaluation context, not an automatic sampler configuration. Explicit
    sampling behavior belongs to the supplied probly representer. Keep the model and its
    parameters unchanged while games use it.

    Args:
        model: A Torch ``nn.Module`` whose parameters and buffers are all on CPU.

    Returns:
        A fresh context-manager factory accepting an integer seed. Pass it as
        ``ProblyPredictor(..., evaluation_context=torch_cpu_context(model))``.

    Raises:
        ImportError: If Torch is unavailable.
        TypeError: If ``model`` is not a Torch module.
        ValueError: If any parameter or buffer is on an accelerator.
    """
    torch = require("torch", purpose="the CPU model evaluation context", extra="probly-torch")
    if not isinstance(model, torch.nn.Module):
        msg = "torch_cpu_context requires a torch.nn.Module."
        raise TypeError(msg)

    def check_cpu() -> None:
        if any(tensor.device.type != "cpu" for tensor in (*model.parameters(), *model.buffers())):
            msg = "torch_cpu_context supports CPU models only; move all parameters and buffers to CPU."
            raise ValueError(msg)

    check_cpu()

    @contextmanager
    def context(seed: int) -> Iterator[None]:
        check_cpu()
        flags = [(module, module.training) for module in model.modules()]
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            torch.random.set_rng_state(generator.get_state())
            try:
                model.eval()
                yield
            finally:
                for module, training in flags:
                    module.training = training

    return context
