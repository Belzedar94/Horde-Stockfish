#!/bin/sh

sha256sum=$( (command -v shasum >/dev/null 2>&1 && echo "shasum -a 256") ||
  (command -v sha256sum >/dev/null 2>&1 && echo "sha256sum"))

if [ -z "$sha256sum" ]; then
  >&2 echo "sha256sum not found, NNUE files will be assumed valid."
fi

get_nnue_filename() {
  sed -n "s|.*#define[[:space:]]*$1[[:space:]]*\"\([^\"]*\.nnue\)\".*|\1|p" evaluate.h
}

validate_network() {
  expected="b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7"
  [ -n "$sha256sum" ] || return 1
  [ -f "$1" ] || return 1
  actual="$($sha256sum "$1" | cut -d ' ' -f 1 | tr '[:upper:]' '[:lower:]')"
  [ "$actual" = "$expected" ]
}

fetch_network() {
  _filename="$(get_nnue_filename "$1")"

  if [ -z "$_filename" ]; then
    >&2 echo "NNUE file name not found for: $1"
    return 1
  fi

  if validate_network "$_filename"; then
    echo "Canonical Run 6B network validated: $_filename"
    return
  fi

  >&2 echo "Canonical Run 6B network is missing or has the wrong SHA-256: $_filename"
  return 1
}

fetch_network EvalFileDefaultName

if [ "$1" = "0" ]; then
    DUMP_FILE=universal/network_dump.inc
    echo -n '"' > $DUMP_FILE
    hexdump -v -e '"\\" "x" 1/1 "%02X"' "$(get_nnue_filename EvalFileDefaultName)" >> $DUMP_FILE
    echo -n '"' >> $DUMP_FILE
fi
