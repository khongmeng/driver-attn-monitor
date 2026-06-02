#!/usr/bin/env bash
# Install a DMS Recorder launcher icon on the user's Desktop.
# Run once after cloning the repo. Re-running overwrites the launcher.
#
# Resolves the repo path from this script's location, so it works no matter
# where the repo is cloned (no hardcoded /home/<user>/... paths).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
DESKTOP_FILE="$DESKTOP_DIR/DMS-Recorder.desktop"

mkdir -p "$DESKTOP_DIR"
chmod +x "$SCRIPT_DIR/start_recording.sh"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DMS Recorder
Comment=Driver Monitoring System data-collection recorder
Exec=$SCRIPT_DIR/start_recording.sh
Path=$REPO_DIR
Icon=camera-video
Terminal=true
Categories=Utility;AudioVideo;Recorder;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

# GNOME trust flag — required on Nautilus 42+ to launch a .desktop file on
# double-click (otherwise it opens as text). Trust metadata is per-user
# state (~/.local/share/gvfs-metadata/), not carried by git, so cloners
# must run this script once.
if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

echo "Installed launcher: $DESKTOP_FILE"
echo "Double-click the icon on your Desktop to launch the recorder."
