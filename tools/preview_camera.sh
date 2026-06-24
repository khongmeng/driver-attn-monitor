#!/usr/bin/env bash
# Live preview for the USB (Logitech C922) webcam.
#
# Cheese is broken on Jetson (libjpeg ABI mismatch), so this pulls the camera's
# MJPEG stream and decodes it with GStreamer's jpegdec instead. Double-click in
# the file manager (right-click -> "Run as a Program") or run from a terminal.
#
# Usage: ./preview_camera.sh [/dev/videoN]   (defaults to /dev/video1)
set -euo pipefail

DEVICE="${1:-/dev/video1}"

if [ ! -e "$DEVICE" ]; then
    echo "No such device: $DEVICE" >&2
    echo "Available cameras:" >&2
    v4l2-ctl --list-devices 2>/dev/null >&2 || ls /dev/video* >&2
    read -rp "Press Enter to close..." _
    exit 1
fi

echo "Previewing $DEVICE  (close the window or press Ctrl+C to stop)"
# nv3dsink is the Jetson-native video sink; autovideosink often renders nothing
# on the Orin. Fall back to xvimagesink if nv3dsink is unavailable.
SINK="nv3dsink"
gst-inspect-1.0 nv3dsink >/dev/null 2>&1 || SINK="xvimagesink"
gst-launch-1.0 v4l2src device="$DEVICE" \
    ! image/jpeg,width=1280,height=720,framerate=30/1 \
    ! jpegdec ! videoconvert ! "$SINK" sync=false
