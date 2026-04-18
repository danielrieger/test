#!/bin/bash

# Safe Synchronization Script (WSL -> Windows)
# -------------------------------------------
# This script aligns the Windows workspace with the WSL environment
# while protecting unique data directories from deletion.

SOURCE="/home/daniel/Thesis/smlm_score/"
TARGET="/mnt/c/Users/User/OneDrive/Desktop/Thesis/smlm_score/"

echo "--- Starting Safe Synchronization (Source: WSL) ---"

# Identity sync for code, excludes large data and caches
rsync -avz \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.rmf3' \
    --exclude='*.log' \
    --exclude='examples/info/' \
    --exclude='examples/picked_particles/' \
    --exclude='examples/PDB_Data/' \
    --exclude='examples/ShareLoc_Data/' \
    --exclude='examples/SuReSim_Input/' \
    "$SOURCE" "$TARGET"

echo "--- Sync Complete. Data directories were protected. ---"
echo "To update the git state without losing local data, run:"
echo "cd $TARGET && git fetch origin && git merge origin/master"
