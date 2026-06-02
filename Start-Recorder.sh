#!/usr/bin/env bash
# Click-to-launch wrapper for the DMS Recorder.
#
# In GNOME Files (Nautilus): right-click this file → "Run as a Program".
# After the first time, a normal double-click works too on most setups.
# Or just run it from a terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/tools/start_recording.sh" "$@"
