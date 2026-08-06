/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef DATA_TRAINING_DATA_GENERATOR_H_INCLUDED
#define DATA_TRAINING_DATA_GENERATOR_H_INCLUDED

#include <iosfwd>

namespace Stockfish {

class Engine;

namespace Data {

// Parse and execute the Horde-only self-play generator command. This function
// is linked only into the isolated data-generator executable.
bool generate_training_data(Engine& engine, std::istream& input);

}  // namespace Data
}  // namespace Stockfish

#endif  // DATA_TRAINING_DATA_GENERATOR_H_INCLUDED
