/*
  Horde-Stockfish, a UCI Horde chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef GENFENS_H_INCLUDED
#define GENFENS_H_INCLUDED

#include <iosfwd>
#include <optional>
#include <string>

namespace Stockfish::GenFens {

// Implements the OpenBench opening-generator contract. On success exactly N
// physical-P Horde FENs are written as "info string genfens <fen>" lines.
std::optional<std::string> run(std::istream& args, std::ostream& output);

}  // namespace Stockfish::GenFens

#endif  // #ifndef GENFENS_H_INCLUDED
