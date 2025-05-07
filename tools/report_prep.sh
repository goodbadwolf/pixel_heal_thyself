#!/bin/bash

# Set source directory (where your run folders are located)
# Replace with your actual source directory
SOURCE_DIR=$1

# Set destination directory (where you want to copy the renamed files)
# Replace with your actual destination directory
DEST_DIR=$2

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all run_* folders
for RUN_DIR in "$SOURCE_DIR"/run_*; do
    # Extract run folder name (e.g., run_001)
    RUN_NAME=$(basename "$RUN_DIR")

    echo "Processing $RUN_NAME..."

    # Copy and rename evaluation.txt
    if [ -f "$RUN_DIR/evaluation.txt" ]; then
        cp "$RUN_DIR/evaluation.txt" "$DEST_DIR/${RUN_NAME}_evaluation.txt"
        echo "  Copied evaluation.txt to ${RUN_NAME}_evaluation.txt"
    else
        echo "  WARNING: evaluation.txt not found in $RUN_NAME"
    fi

    # Copy and rename train_loss.txt
    if [ -f "$RUN_DIR/train_loss.txt" ]; then
        cp "$RUN_DIR/train_loss.txt" "$DEST_DIR/${RUN_NAME}_train_loss.txt"
        echo "  Copied train_loss.txt to ${RUN_NAME}_train_loss.txt"
    else
        echo "  WARNING: train_loss.txt not found in $RUN_NAME"
    fi
done

echo "Done! All files have been copied and renamed to $DEST_DIR"
