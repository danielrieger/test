#!/bin/bash

# Safe Synchronization Script (Windows -> WSL)
# -------------------------------------------
# This script aligns the WSL environment with the Windows Deskstop workspace
# while protecting unique data directories from deletion.

SOURCE="/mnt/c/Users/User/OneDrive/Desktop/Thesis/smlm_score/"
TARGET="/home/daniel/Thesis/smlm_score/"

echo "--- Starting Safe Synchronization (Source: Windows) ---"

# Identity sync for code, excludes large data and caches
# NOTE: We DO NOT use --delete for the examples/ directory to ensure 
# that untracked local data (like EMAN2 results) is preserved.

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
