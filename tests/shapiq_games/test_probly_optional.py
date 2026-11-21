"""Core imports remain usable when probly and neural-network backends are absent."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_public_imports_do_not_require_probly_torch_or_jax():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingOptionalPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'probly', 'torch', 'jax', 'flax'}:
                    raise ModuleNotFoundError(fullname, name=fullname)
                return None

        sys.meta_path.insert(0, MissingOptionalPackages())
        import numpy as np
        import shapiq
        import shapiq_games
        from shapiq_games import ImageProblyGame, ProblyGame, ProblyPredictor, torch_cpu_context

        assert not any(name.split('.')[0] in {'probly', 'torch', 'jax', 'flax'} for name in sys.modules)
        try:
            ProblyPredictor(lambda x: x)(np.zeros((1, 1)))
        except ImportError as error:
            assert 'probly' in str(error).lower(), str(error)
        else:
            raise AssertionError('A default probly predictor must explain its missing dependency')
        """
    )
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
