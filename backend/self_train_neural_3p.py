from __future__ import annotations

"""Backward-compatibility shim for the v2 trainer module.

Use ``backend.self_train_neural_3p_v2`` as the canonical import and CLI path.
"""

try:
    from .self_train_neural_3p_v2 import *  # noqa: F401,F403
    from .self_train_neural_3p_v2 import main
except ImportError:
    from self_train_neural_3p_v2 import *  # type: ignore # noqa: F401,F403
    from self_train_neural_3p_v2 import main  # type: ignore


if __name__ == "__main__":
    main()
