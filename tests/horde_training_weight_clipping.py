#!/usr/bin/env python3
"""Every serialized weight stays inside the range its dtype can represent.

The dense layers were clipped into their int8 range from the first campaign.
The feature transformer quantises to int16 and was not clipped at all, so
nothing held it inside the range the exporter then demands it fit. Corpus A
walked 2 weights of 917,504 outside it and the export refused, which is the
right failure but three hours too late.

The bound is per architecture because the scales differ by 64x: legacy
quantises ft_weights at 127, the V2 container and V3 at 127 * 64.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import horde_training_control as ctl  # noqa: E402


class _Stub(nn.Module):
    """The four serialized tensors the clip touches, nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden0_weights = nn.Parameter(torch.zeros(4))
        self.hidden1_weights = nn.Parameter(torch.zeros(4))
        self.output_weights = nn.Parameter(torch.zeros(4))
        self.ft_weights = nn.Parameter(torch.zeros(6))


def test_feature_transformer_is_clipped_to_its_own_scale() -> None:
    for architecture, scale in (
        (ctl.LEGACY_ARCHITECTURE, 127.0),
        ("v3-g1024-pawn-wpc8", 127.0 * 64.0),
    ):
        limit = 32767.0 / scale
        model = _Stub()
        with torch.no_grad():
            # Straddle the edge: outside both ways, exactly on it, well inside.
            model.ft_weights.copy_(torch.tensor([
                limit * 1.05, -limit * 1.05, limit, -limit, limit * 0.5, 0.0,
            ], dtype=torch.float32))
        ctl._clip_serialized_weights(model, architecture)
        values = model.ft_weights.detach()

        if float(values.max()) > limit + 1e-6:
            raise AssertionError(
                f"{architecture}: {float(values.max())} exceeds +{limit}")
        if float(values.min()) < -limit - 1e-6:
            raise AssertionError(
                f"{architecture}: {float(values.min())} exceeds -{limit}")
        # The overshooting pair lands ON the edge, not zeroed and not rescaled.
        if abs(float(values[0]) - limit) > 1e-6:
            raise AssertionError(
                f"{architecture}: positive overshoot not clipped to the edge")
        if abs(float(values[1]) + limit) > 1e-6:
            raise AssertionError(
                f"{architecture}: negative overshoot not clipped to the edge")
        # Anything already inside is untouched.
        if abs(float(values[4]) - limit * 0.5) > 1e-6:
            raise AssertionError(f"{architecture}: an in-range weight moved")

        # And the clipped edge survives the exporter's own rounding, which is
        # the property that actually matters: the export must stop refusing.
        rounded = torch.round(values.to(torch.float64) * scale)
        if int(rounded.max()) > 32767 or int(rounded.min()) < -32768:
            raise AssertionError(
                f"{architecture}: clipped weights still overflow i16 after "
                f"rounding: [{int(rounded.min())}, {int(rounded.max())}]")


def test_the_dense_clipping_still_holds() -> None:
    model = _Stub()
    dense_limit = 127.0 / 64.0
    output_limit = (127.0 * 127.0) / 9600.0
    with torch.no_grad():
        model.hidden0_weights.fill_(dense_limit * 3)
        model.hidden1_weights.fill_(-dense_limit * 3)
        model.output_weights.fill_(output_limit * 3)
    ctl._clip_serialized_weights(model, ctl.LEGACY_ARCHITECTURE)
    if abs(float(model.hidden0_weights.max()) - dense_limit) > 1e-6:
        raise AssertionError("hidden0 clipping regressed")
    if abs(float(model.hidden1_weights.min()) + dense_limit) > 1e-6:
        raise AssertionError("hidden1 clipping regressed")
    if abs(float(model.output_weights.max()) - output_limit) > 1e-6:
        raise AssertionError("output clipping regressed")


def main() -> int:
    test_feature_transformer_is_clipped_to_its_own_scale()
    test_the_dense_clipping_still_holds()
    print("Horde serialized weight clipping: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
