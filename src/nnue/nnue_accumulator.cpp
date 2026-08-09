/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "nnue_accumulator.h"

#include <cassert>

namespace Stockfish::Eval::NNUE {

const AccumulatorState& AccumulatorStack::latest() const noexcept {
    assert(size > 0);
    return accumulators[size - 1];
}

AccumulatorState& AccumulatorStack::mut_latest() noexcept {
    assert(size > 0);
    return accumulators[size - 1];
}

void AccumulatorStack::reset() noexcept {
    accumulators[0].computed.fill(false);
    accumulators[0].dirtyPiece = {};
    size                       = 1;
}

DirtyPiece& AccumulatorStack::push() noexcept {
    assert(size < MaxSize);
    AccumulatorState& next = accumulators[size++];
    next.computed.fill(false);
    next.dirtyPiece = {};
    return next.dirtyPiece;
}

void AccumulatorStack::pop() noexcept {
    assert(size > 1);
    --size;
}

}  // namespace Stockfish::Eval::NNUE
