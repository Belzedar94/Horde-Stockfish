/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef DATA_TRAINING_DATA_H_INCLUDED
#define DATA_TRAINING_DATA_H_INCLUDED

#include <string>
#include <utility>

#include "types.h"

namespace Stockfish::Data {

enum class HordeOutcomeReason : u8 {
    CHECKMATE           = 1,
    EXTINCTION          = 2,
    STALEMATE           = 3,
    HORDE_FORTRESS      = 4,
    FIFTY_MOVE          = 5,
    FIVEFOLD_REPETITION = 6
};

struct TrainingDataSample {
    std::string        fen;
    int                score         = 0;
    Move               bestMove      = Move::none();
    Move               playedMove    = Move::none();
    int                result        = 0;
    HordeOutcomeReason outcomeReason = HordeOutcomeReason::STALEMATE;
};

enum class DataError {
    NONE,
    UNSUPPORTED_BYTE_ORDER,
    UNSUPPORTED_POSITION,
    INVALID_MOVE,
    SCORE_OUT_OF_RANGE,
    RESULT_OUT_OF_RANGE,
    OUTCOME_OUT_OF_RANGE,
    POSITION_CLOCK_OUT_OF_RANGE,
    INVALID_MANIFEST,
    OUTPUT_EXISTS,
    OPEN_FAILED,
    WRITE_FAILED,
    SEEK_FAILED,
    CLOSE_FAILED,
    EMPTY_DATASET,
    RECORD_COUNT_MISMATCH,
    ABORT_FAILED,
    SINK_CLOSED
};

struct DataResult {
    DataError   error = DataError::NONE;
    std::string message;

    explicit operator bool() const { return error == DataError::NONE; }

    static DataResult success() { return {}; }
    static DataResult failure(DataError failureError, std::string message) {
        return {failureError, std::move(message)};
    }
};

class DatasetSink {
   public:
    virtual ~DatasetSink() = default;

    virtual DataResult append(const TrainingDataSample& sample) = 0;
    virtual DataResult finalize()                               = 0;
    virtual DataResult abort()                                  = 0;
};

}  // namespace Stockfish::Data

#endif  // DATA_TRAINING_DATA_H_INCLUDED
