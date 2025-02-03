#!/usr/bin/env bash

# SOURCE is your new/modified builder.py (relative to the script's location)
SOURCE="$(dirname "$0")/builder.py"

# Find the TARGET location dynamically (inside the Docker container)
TARGET=$(find /app -path "*/onnxruntime_genai/models/builder.py" -type f -print -quit)

if [ -z "$TARGET" ]; then
    echo "Error: builder.py not found in /app!"
    exit 1
fi

# OPTIONAL: Backup the current builder.py just in case
BACKUP="${TARGET}.bak"
if [ -f "$TARGET" ]; then
    echo "Backing up $TARGET to $BACKUP ..."
    cp "$TARGET" "$BACKUP"
else
    echo "Error: $TARGET does not exist!"
    exit 1
fi

# Overwrite the target with the source
echo "Overwriting $TARGET with $SOURCE ..."
cp "$SOURCE" "$TARGET"

if [ $? -eq 0 ]; then
    echo "Done. $TARGET has been replaced."
else
    echo "Error: Failed to replace $TARGET!"
    exit 1
fi