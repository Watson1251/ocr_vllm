#!/bin/bash

# Usage:
#   ./watch_image.sh [-z] [-f] /path/to/image
#
#   -z  enable auto-zoom to fit window on first load (feh -Z)
#   -f  fullscreen mode (feh -F)

ZOOM_FLAG=""
FULL_FLAG=""

# Parse flags
while getopts "zf" opt; do
  case "$opt" in
    z) ZOOM_FLAG="-Z" ;;
    f) FULL_FLAG="-F" ;;
  esac
done

shift $((OPTIND - 1))
IMG="$1"
INTERVAL=0.5

if [ -z "$IMG" ]; then
  echo "Usage: $0 [-z] [-f] /path/to/image"
  exit 1
fi

echo "Watching: $IMG"

# Wait for file to exist
while [ ! -f "$IMG" ]; do
  echo "[ Waiting for $IMG to be created ... ]"
  sleep "$INTERVAL"
done

echo "Launching viewer: feh --reload 0 -Z $FULL_FLAG \"$IMG\""

# IMPORTANT: start feh directly so $PID is the feh process
feh --reload 0 -Z $FULL_FLAG "$IMG" &
PID=$!

# Initial modification time
last_mtime=$(stat -c %Y "$IMG" 2>/dev/null || echo 0)

# Watch for changes
while kill -0 "$PID" 2>/dev/null; do
  sleep "$INTERVAL"

  # If file is temporarily missing, just wait
  if [ ! -f "$IMG" ]; then
    continue
  fi

  mtime=$(stat -c %Y "$IMG" 2>/dev/null || echo 0)

  if [ "$mtime" != "$last_mtime" ]; then
    echo "[ Change detected — reloading viewer ]"
    last_mtime="$mtime"
    # Reload current image in single-image slideshow / multiwindow mode
    kill -USR1 "$PID"
  fi
done

echo "Viewer closed. Exiting watcher."
