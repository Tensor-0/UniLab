"""TEMPORARY LOCAL EXPERIMENT SHIM — 2026-09-20, unisim-core 1.1.4 -> 1.7.2.

`unisim.dr.types` dropped `GeomSizeOverride`, `InitRandomizationPlan` and
`ModelVariantSpec` when the per-environment model plan moved from "override geom
sizes inside one model" to "select among whole model sources"
(`FixedVariantPlan` + `ModelSourceDescriptor`). The three names are used by
`unilab.tasks.manipulation.sharpa_inhand.rotation` and by an annotation in
`unilab/dr/provider.py`; nothing on the dm10 locomotion path touches them.

These placeholders exist ONLY so `ensure_registries()` can import every task
module during a throughput measurement. They raise on use, deliberately — a
silent no-op would let a sharpa run proceed with no randomization at all.

Delete this module and migrate the two call sites properly before keeping the
dependency bump.
"""

from __future__ import annotations

__all__ = ["GeomSizeOverride", "InitRandomizationPlan", "ModelVariantSpec"]

try:  # pragma: no cover - experiment shim
    from unisim.dr.types import (
        GeomSizeOverride,
        InitRandomizationPlan,
        ModelVariantSpec,
    )
except ImportError:  # pragma: no cover - experiment shim

    class _RemovedInNewUnisim:
        _name = "?"

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise NotImplementedError(
                f"unisim.dr.types.{type(self)._name} was removed in unisim-core >= 1.4.2 "
                "(replaced by FixedVariantPlan / ModelSourceDescriptor). This placeholder "
                "only exists so task modules import during a local dependency experiment; "
                "see unilab/dr/_compat.py."
            )

    class GeomSizeOverride(_RemovedInNewUnisim):  # type: ignore[no-redef]
        _name = "GeomSizeOverride"

    class InitRandomizationPlan(_RemovedInNewUnisim):  # type: ignore[no-redef]
        _name = "InitRandomizationPlan"

    class ModelVariantSpec(_RemovedInNewUnisim):  # type: ignore[no-redef]
        _name = "ModelVariantSpec"
