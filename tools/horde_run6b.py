#!/usr/bin/env python3
"""Fail-closed trainer-side reader and integer replay for Horde Run 6B."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import sys
from typing import Sequence

try:
    from .horde_training_decoder import (
        BLACK,
        LEGACY_DIMENSIONS,
        SparseFeatures,
        WHITE,
    )
except ImportError:
    from horde_training_decoder import BLACK, LEGACY_DIMENSIONS, SparseFeatures, WHITE


RUN6B_SHA256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7"
FILE_SIZE = 1_088_416
FILE_VERSION = 0x7AF32F20
NETWORK_HASH = 0x3C103E72
TRANSFORMER_HASH = 0x5F2348B8
ARCHITECTURE_HASH = 0x633376CA

ACCUMULATOR_DIMENSIONS = 512
PSQT_BUCKETS = 8
LAYER_STACKS = 8
NETWORK_INPUTS = 2 * ACCUMULATOR_DIMENSIONS
WEIGHT_SCALE_BITS = 6
START_PIECE_COUNT = 52


class Run6BError(ValueError):
    """Raised when the pinned network or replay inputs violate the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Run6BError(message)


@dataclass(frozen=True, slots=True)
class DenseLayer:
    input_dimensions: int
    padded_input_dimensions: int
    output_dimensions: int
    biases: array
    weights: array


@dataclass(frozen=True, slots=True)
class LayerStack:
    fc0: DenseLayer
    fc1: DenseLayer
    fc2: DenseLayer


@dataclass(frozen=True, slots=True)
class RawOutput:
    psqt: int
    positional: int

    @property
    def total(self) -> int:
        return self.psqt + self.positional


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def u32(self) -> int:
        _require(self.offset + 4 <= len(self.payload), "Run 6B header is truncated")
        value = struct.unpack_from("<I", self.payload, self.offset)[0]
        self.offset += 4
        return value

    def text(self, size: int) -> str:
        _require(0 <= size <= 1 << 20, f"invalid Run 6B description length {size}")
        _require(self.offset + size <= len(self.payload), "Run 6B description is truncated")
        raw = self.payload[self.offset : self.offset + size]
        self.offset += size
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Run6BError(f"Run 6B description is not UTF-8: {error}") from error

    def values(self, typecode: str, count: int, label: str) -> array:
        item_sizes = {"b": 1, "h": 2, "i": 4}
        _require(typecode in item_sizes, f"unsupported array type {typecode}")
        size = item_sizes[typecode] * count
        _require(self.offset + size <= len(self.payload), f"{label} is truncated")
        values = array(typecode)
        values.frombytes(self.payload[self.offset : self.offset + size])
        self.offset += size
        _require(values.itemsize == item_sizes[typecode], f"host {label} item size is unsupported")
        if sys.byteorder != "little" and values.itemsize > 1:
            values.byteswap()
        return values


def _read_dense(reader: _Reader, inputs: int, padded_inputs: int, outputs: int) -> DenseLayer:
    biases = reader.values("i", outputs, "dense biases")
    weights = reader.values("b", outputs * padded_inputs, "dense weights")
    return DenseLayer(inputs, padded_inputs, outputs, biases, weights)


def _signed_wrap(value: int, bits: int) -> int:
    modulus = 1 << bits
    value &= modulus - 1
    return value - modulus if value & (1 << (bits - 1)) else value


def _trunc_div2(value: int) -> int:
    return value // 2 if value >= 0 else -((-value) // 2)


def _activate(value: int) -> int:
    return max(0, min(value >> WEIGHT_SCALE_BITS, 127))


class Run6BNetwork:
    def __init__(
        self,
        description: str,
        biases: array,
        weights: array,
        psqt_weights: array,
        layers: tuple[LayerStack, ...],
    ) -> None:
        self.description = description
        self.biases = biases
        self.weights = weights
        self.psqt_weights = psqt_weights
        self.layers = layers

    @classmethod
    def load(cls, path: Path) -> Run6BNetwork:
        return cls.load_registered(path, RUN6B_SHA256, FILE_SIZE, "Run 6B")

    @classmethod
    def load_registered(
        cls,
        path: Path,
        expected_sha256: str,
        expected_size: int,
        artifact_name: str = "registered legacy NNUE",
    ) -> Run6BNetwork:
        resolved = path.expanduser().resolve()
        payload = resolved.read_bytes()
        _require(
            isinstance(expected_size, int) and expected_size > 0,
            "registered legacy NNUE size is invalid",
        )
        expected_sha256 = expected_sha256.upper()
        _require(
            len(expected_sha256) == 64
            and all(character in "0123456789ABCDEF" for character in expected_sha256),
            "registered legacy NNUE SHA-256 is invalid",
        )
        _require(
            len(payload) == expected_size,
            f"{artifact_name} size mismatch: {len(payload)}",
        )
        observed_sha = hashlib.sha256(payload).hexdigest().upper()
        _require(
            observed_sha == expected_sha256,
            f"{artifact_name} SHA-256 mismatch: {observed_sha}",
        )

        reader = _Reader(payload)
        version = reader.u32()
        network_hash = reader.u32()
        description = reader.text(reader.u32())
        transformer_hash = reader.u32()
        _require(version == FILE_VERSION, f"{artifact_name} version mismatch: 0x{version:08X}")
        _require(
            network_hash == NETWORK_HASH,
            f"{artifact_name} network hash mismatch: 0x{network_hash:08X}",
        )
        _require(
            transformer_hash == TRANSFORMER_HASH,
            f"{artifact_name} transformer hash mismatch: 0x{transformer_hash:08X}",
        )

        biases = reader.values("h", ACCUMULATOR_DIMENSIONS, "feature-transformer biases")
        weights = reader.values(
            "h",
            LEGACY_DIMENSIONS * ACCUMULATOR_DIMENSIONS,
            "feature-transformer weights",
        )
        psqt_weights = reader.values(
            "i", LEGACY_DIMENSIONS * PSQT_BUCKETS, "PSQT weights"
        )

        layers: list[LayerStack] = []
        for stack_index in range(LAYER_STACKS):
            architecture_hash = reader.u32()
            _require(
                architecture_hash == ARCHITECTURE_HASH,
                f"{artifact_name} stack {stack_index} architecture hash mismatch: "
                f"0x{architecture_hash:08X}",
            )
            layers.append(
                LayerStack(
                    _read_dense(reader, NETWORK_INPUTS, NETWORK_INPUTS, 16),
                    _read_dense(reader, 16, 32, 32),
                    _read_dense(reader, 32, 32, 1),
                )
            )
        _require(reader.offset == len(payload), f"{artifact_name} contains trailing bytes")
        return cls(description, biases, weights, psqt_weights, tuple(layers))

    def _accumulate(self, indices: Sequence[int], bucket: int) -> tuple[list[int], int]:
        _require(all(0 <= index < LEGACY_DIMENSIONS for index in indices),
                 "legacy replay received an out-of-range feature")
        accumulator = [int(value) for value in self.biases]
        psqt = 0
        for feature in indices:
            weight_offset = feature * ACCUMULATOR_DIMENSIONS
            for lane in range(ACCUMULATOR_DIMENSIONS):
                accumulator[lane] += self.weights[weight_offset + lane]
            psqt += self.psqt_weights[feature * PSQT_BUCKETS + bucket]
        return ([_signed_wrap(value, 16) for value in accumulator], _signed_wrap(psqt, 32))

    @staticmethod
    def _dense(layer: DenseLayer, inputs: Sequence[int]) -> list[int]:
        _require(len(inputs) == layer.input_dimensions, "dense replay input size mismatch")
        outputs: list[int] = []
        for output in range(layer.output_dimensions):
            value = int(layer.biases[output])
            offset = output * layer.padded_input_dimensions
            for input_index, input_value in enumerate(inputs):
                value += layer.weights[offset + input_index] * input_value
            outputs.append(value)
        return outputs

    def evaluate(self, features: SparseFeatures, side_to_move: int) -> RawOutput:
        _require(side_to_move in (WHITE, BLACK), f"invalid side to move {side_to_move}")
        piece_count = len(features.v2_global)
        _require(1 <= piece_count <= START_PIECE_COUNT, f"invalid replay piece count {piece_count}")
        _require(
            len(features.legacy_white) == len(features.legacy_black) == piece_count,
            "legacy replay feature counts differ",
        )
        bucket = max(0, min((piece_count - 1) * PSQT_BUCKETS // START_PIECE_COUNT,
                            PSQT_BUCKETS - 1))

        white_accumulator, white_psqt = self._accumulate(features.legacy_white, bucket)
        black_accumulator, black_psqt = self._accumulate(features.legacy_black, bucket)
        if side_to_move == WHITE:
            us_accumulator, them_accumulator = white_accumulator, black_accumulator
            us_psqt, them_psqt = white_psqt, black_psqt
        else:
            us_accumulator, them_accumulator = black_accumulator, white_accumulator
            us_psqt, them_psqt = black_psqt, white_psqt

        transformed = [max(0, min(value, 127)) for value in us_accumulator]
        transformed.extend(max(0, min(value, 127)) for value in them_accumulator)
        layer = self.layers[bucket]
        hidden0 = [_activate(value) for value in self._dense(layer.fc0, transformed)]
        hidden1 = [_activate(value) for value in self._dense(layer.fc1, hidden0)]
        positional = self._dense(layer.fc2, hidden1)[0]
        return RawOutput(_trunc_div2(us_psqt - them_psqt), positional)


__all__ = ["RawOutput", "Run6BError", "Run6BNetwork"]
