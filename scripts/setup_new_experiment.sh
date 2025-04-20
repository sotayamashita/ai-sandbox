#!/bin/bash
#
# Script to create a new experiment directory with Python virtual environment.
#
# Usage:
#   ./scripts/setup_experiments.sh <new_experiment_name>
#
set -e

readonly NEW_EXPERIMENT_NAME="$1"
readonly NEW_EXPERIMENT_DIR="experiments/${NEW_EXPERIMENT_NAME}"

# Check if the experiment name is provided
if [[ -z "${NEW_EXPERIMENT_NAME}" ]]; then
    echo "Usage: $0 <new_experiment_name>" >&2
    exit 1
fi

# Check if the experiment directory already exists
if [[ -d "${NEW_EXPERIMENT_DIR}" ]]; then
    echo "Experiment directory already exists: ${NEW_EXPERIMENT_DIR}" >&2
    exit 1
fi

# Create the experiment directory
mkdir -p "${NEW_EXPERIMENT_DIR}"

# Create the .python-version file
echo "3.12" > "${NEW_EXPERIMENT_DIR}/.python-version"

# Call `deactivate` to deactivate the current virtual environment
deactivate || true

# Create a new virtual environment
uv venv "${NEW_EXPERIMENT_DIR}/.venv"

# Activate the new virtual environment
source "${NEW_EXPERIMENT_DIR}/.venv/bin/activate.fish"

echo "cd ${NEW_EXPERIMENT_DIR}"
echo "Happy hacking!"
