#!/bin/bash
# Verify the canonical Fairy-Stockfish Horde perft vectors.

TESTS_FAILED=0

error()
{
  echo "Horde perft testing failed on line $1"
  exit 1
}
trap 'error ${LINENO}' ERR

echo "Horde perft testing started"

EXPECT_SCRIPT=$(mktemp)

cat << 'EOF' > "$EXPECT_SCRIPT"
#!/usr/bin/expect -f
set timeout 120
lassign [lrange $argv 0 3] pos depth result logfile
log_file -noappend $logfile
spawn ./stockfish
send "position $pos\ngo perft $depth\n"
expect {
  "Nodes searched: $result" {}
  timeout {puts "TIMEOUT: Expected $result nodes"; exit 1}
  eof {puts "EOF: Horde-Stockfish crashed"; exit 2}
}
send "quit\n"
expect eof
EOF

chmod +x "$EXPECT_SCRIPT"

run_test() {
  local pos="$1"
  local depth="$2"
  local expected="$3"
  local tmp_file
  tmp_file=$(mktemp)

  echo -n "Testing depth $depth: ${pos:0:50}... "

  if "$EXPECT_SCRIPT" "$pos" "$depth" "$expected" "$tmp_file" > /dev/null 2>&1; then
    echo "OK"
    rm -f "$tmp_file"
  else
    local exit_code=$?
    echo "FAILED (exit code: $exit_code)"
    echo "===== Output for failed test ====="
    cat "$tmp_file"
    echo "=================================="
    rm -f "$tmp_file"
    TESTS_FAILED=1
  fi
}

run_test "startpos" 4 23310
run_test "fen 4k3/pp4q1/3P2p1/8/P3PP2/PPP2r2/PPP5/PPPP4 b - - 0 1" 4 56539
run_test "fen k7/5p2/4p2P/3p2P1/2p2P2/1p2P2P/p2P2P1/2P2P2 w - - 0 1" 4 33781
run_test "fen 4k3/7r/8/P7/2p1n2P/3p2P1/1P3P2/PPP1PPP1 w - - 0 1" 4 128809
run_test "fen rnbqkbnr/6p1/2p1Pp1P/P1PPPP2/Pp4PP/1p2PPPP/1P2PPPP/PP1nPPPP b kq a3 0 18" 4 197287

rm -f "$EXPECT_SCRIPT"
echo "Horde perft testing completed"

if [ "$TESTS_FAILED" -ne 0 ]; then
  echo "Some Horde perft tests failed"
  exit 1
fi
