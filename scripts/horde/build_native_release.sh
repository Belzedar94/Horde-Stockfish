#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "usage: $0 PLATFORM ARCH VERSION SOURCE_DATE_EPOCH COMMIT" >&2
    echo "  PLATFORM: linux | windows" >&2
    echo "  ARCH: x86-64-avx2 | x86-64-bmi2" >&2
    exit 2
}

die() {
    echo "build_native_release: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

[[ $# -eq 5 ]] || usage

platform=$1
architecture=$2
version=$3
source_date_epoch=$4
commit=$5

case "$platform" in
    linux | windows) ;;
    *) usage ;;
esac

case "$architecture" in
    x86-64-avx2 | x86-64-bmi2) ;;
    *) usage ;;
esac

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "VERSION must be a semantic x.y.z version"
[[ "$source_date_epoch" =~ ^(0|[1-9][0-9]*)$ ]] \
    || die "SOURCE_DATE_EPOCH must be canonical non-negative decimal"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] \
    || die "COMMIT must be one full lowercase SHA-1"

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)
[[ "$(pwd -P)" == "$repo_root" ]] \
    || die "run this recipe from the repository root: $repo_root"

required_files=(
    AUTHORS
    CITATION.cff
    Copying.txt
    README.md
    docs/horde/baseline-manifest.json
    docs/horde/release-notes-draft.md
    docs/horde/release-process.md
    docs/horde/testing-and-release-contract.md
    networks/CC0-1.0-NOTICE.md
    networks/README.md
    src/Makefile
    src/evaluate.h
)
for path in "${required_files[@]}"; do
    [[ -f "$path" && ! -L "$path" ]] || die "missing regular release input: $path"
done

default_net=$(
    sed -n 's/.*EvalFileDefaultName[[:space:]]*"\([^"]*\.nnue\)".*/\1/p' \
        src/evaluate.h | head -n 1
)
case "$default_net" in
    ../networks/*.nnue) ;;
    *) die "default network must resolve through ../networks: $default_net" ;;
esac
network_path=${default_net#../}
[[ -f "$network_path" && ! -L "$network_path" ]] \
    || die "default network is not one tracked regular file: $network_path"

require_command date
require_command find
require_command install
require_command make
require_command sed
require_command touch

[[ ! -e build ]] || die "build/ already exists; use a clean release root"
dirty_native=$(find src -maxdepth 1 \( -type f -o -type l \) \
    \( -name '*.o' -o -name '.build_*.txt' -o -name 'horde-stockfish' \
       -o -name 'horde-stockfish.exe' -o -name 'stockfish' -o -name 'stockfish.exe' \) \
    -print -quit)
[[ -z "$dirty_native" ]] || die "native build product already exists: $dirty_native"

export SOURCE_DATE_EPOCH=$source_date_epoch
export LC_ALL=C
export LANG=C
export TZ=UTC
umask 022

# Caller-provided flags must not change release bytes.
unset CXXFLAGS CPPFLAGS LDFLAGS DEPENDFLAGS EXTRACXXFLAGS EXTRALDFLAGS
unset MAKEFLAGS MFLAGS TAR_OPTIONS XZ_DEFAULTS XZ_OPT ZIPOPT

build_date=$(date -u -d "@$SOURCE_DATE_EPOCH" +%Y%m%d)
short_commit=${commit:0:8}

if [[ "$platform" == linux ]]; then
    require_command g++
    require_command tar
    require_command xz
    [[ "$(g++ -dumpmachine)" == x86_64-linux-gnu ]] \
        || die "Linux release compiler does not target x86_64-linux-gnu"
    tar --version | head -n 1 | grep -Fq 'GNU tar' \
        || die "Linux release packaging requires GNU tar"

    make -C src -j2 ARCH="$architecture" COMP=gcc \
        EXE=horde-stockfish GIT_SHA="$short_commit" GIT_DATE="$build_date" \
        GIT_DIFFINDEX= build
    engine=src/horde-stockfish
    extension=tar.xz
else
    require_command cmake
    [[ -n "${CXX:-}" ]] || die "CXX must name the pinned MinGW compiler"
    [[ "$CXX" == /* && -x "$CXX" ]] \
        || die "CXX must be an executable absolute path: $CXX"
    [[ "$("$CXX" -dumpmachine)" == x86_64-w64-mingw32.static ]] \
        || die "MinGW release compiler does not use the pinned static x86_64 target"
    if [[ ${#source_date_epoch} -gt 10 ]] \
        || (( 10#$source_date_epoch < 315532800 )); then
        die "Windows ZIP epoch must be representable on or after 1980-01-01"
    fi

    make -C src -j2 ARCH="$architecture" COMP=mingw COMPCXX="$CXX" \
        EXE=horde-stockfish.exe GIT_SHA="$short_commit" GIT_DATE="$build_date" \
        GIT_DIFFINDEX= build
    engine=src/horde-stockfish.exe
    extension=zip
fi

[[ -f "$engine" && ! -L "$engine" ]] \
    || die "native engine was not produced as one regular file: $engine"

release_root="Horde-Stockfish-$version"
stage_parent=build/release-stage
stage_root="$stage_parent/$release_root"
release_dir=build/release
asset_name="Horde-Stockfish-$version-$platform-$architecture.$extension"
asset_path="$release_dir/$asset_name"

mkdir -p "$stage_root/bin" "$stage_root/docs" "$stage_root/networks" "$release_dir"
install -m 0755 "$engine" "$stage_root/bin/$(basename -- "$engine")"
install -m 0644 AUTHORS CITATION.cff Copying.txt README.md "$stage_root/"
install -m 0644 docs/horde/baseline-manifest.json \
    "$stage_root/docs/BASELINE_MANIFEST.json"
install -m 0644 docs/horde/release-notes-draft.md \
    "$stage_root/docs/RELEASE_NOTES_DRAFT.md"
install -m 0644 docs/horde/release-process.md \
    "$stage_root/docs/RELEASE_PROCESS.md"
install -m 0644 docs/horde/testing-and-release-contract.md \
    "$stage_root/docs/TESTING_AND_RELEASE_CONTRACT.md"
install -m 0644 networks/CC0-1.0-NOTICE.md networks/README.md "$stage_root/networks/"
# Keep the source/default filename unchanged. The public package alias is
# stable and byte-for-byte identical to the tracked Run 6B input.
install -m 0644 "$network_path" "$stage_root/networks/Horde_v1.nnue"

cat > "$stage_root/SOURCE.md" <<EOF
# Exact source

This Horde-Stockfish $version candidate was built from commit \
\`$commit\`.

Source: https://github.com/Belzedar94/Horde-Stockfish/tree/$commit

Build recipe: \`scripts/horde/build_native_release.sh\`
EOF

if [[ "$platform" == linux ]]; then
    tar --sort=name --format=gnu --mtime="@$SOURCE_DATE_EPOCH" \
        --owner=0 --group=0 --numeric-owner \
        -C "$stage_parent" -cf - "$release_root" \
        | xz --threads=1 -9e --check=crc64 > "$asset_path"
else
    find "$stage_root" -exec touch -d "@$SOURCE_DATE_EPOCH" {} +
    (
        cd "$stage_parent"
        cmake -E tar cf "../release/$asset_name" --format=zip "$release_root"
    )
fi

[[ -s "$asset_path" && ! -L "$asset_path" ]] \
    || die "release archive was not produced as one non-empty regular file"

shopt -s nullglob
release_outputs=("$release_dir"/*)
[[ ${#release_outputs[@]} -eq 1 && "${release_outputs[0]}" == "$asset_path" ]] \
    || die "release recipe produced an unexpected output set"

printf '%s\n' "$asset_path"
