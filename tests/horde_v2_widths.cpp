/*
  Deterministic footprint and work receipts for the experimental Horde V2
  width ladder. Timing and engine NPS are separate, machine-specific gates.
*/

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "../src/nnue/horde_v2_widths.h"

using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

inline constexpr std::size_t StartRoyalRows  = 51;
inline constexpr std::size_t StartGlobalRows = 52;

void require(bool condition, const char* message) {
    if (!condition)
    {
        std::cerr << "Horde V2 width failure: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template<typename Width>
void emit(const char* name,
          std::size_t expectedStartBytes,
          std::size_t expectedQuietBytes,
          std::size_t expectedKingMoveBytes) {
    constexpr std::size_t StartBytes =
      Width::template sparse_weight_bytes<StartRoyalRows, StartGlobalRows>();
    constexpr std::size_t QuietBytes    = Width::template sparse_weight_bytes<2, 2>();
    constexpr std::size_t KingMoveBytes = Width::template sparse_weight_bytes<StartRoyalRows, 2>();

    require(StartBytes == expectedStartBytes, "start refresh byte mismatch");
    require(QuietBytes == expectedQuietBytes, "quiet delta byte mismatch");
    require(KingMoveBytes == expectedKingMoveBytes, "king move byte mismatch");

    std::cout << '{' << "\"name\":\"" << name << "\",\"royal_lanes\":" << Width::RoyalLanes
              << ",\"global_lanes\":" << Width::GlobalLanes
              << ",\"parameter_bytes\":" << Width::ParameterBytes
              << ",\"royal_table_bytes\":" << Width::RoyalTableBytes
              << ",\"global_table_bytes\":" << Width::GlobalTableBytes
              << ",\"accumulator_bytes\":" << Width::AccumulatorBytes
              << ",\"dense_macs\":" << Width::DenseMacs
              << ",\"start_refresh_weight_bytes\":" << StartBytes
              << ",\"quiet_delta_weight_bytes\":" << QuietBytes
              << ",\"king_move_weight_bytes\":" << KingMoveBytes << "}\n";
}

}  // namespace

int main() {
    emit<Width256x256>("256+256", 52736, 2048, 27136);
    emit<Width128x256>("128+256", 39680, 1536, 14080);
    emit<Width128x128>("128+128", 26368, 1024, 13568);
    emit<Width64x192>("64+192", 26496, 1024, 7296);
    std::cout << "Horde V2 width contracts passed\n";
}
