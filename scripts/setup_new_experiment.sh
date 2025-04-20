#!/bin/bash
#
# Script to create a new experiment directory with Python virtual environment.
#
# Usage:
#   ./scripts/setup_new_experiment.sh <new_experiment_name>
#
set -e

readonly NEW_EXPERIMENT_NAME="$1"
readonly NEW_EXPERIMENT_DIR="experiments/${NEW_EXPERIMENT_NAME}"

# Create README.md with appropriate content
function create_readme() {
  local readme_file="${NEW_EXPERIMENT_DIR}/README.md"
  
  echo "# ${NEW_EXPERIMENT_NAME}" > "${readme_file}"
  echo "Created README.md file: ${readme_file}"
  
  read -p "Is this experiment paper based? (y/n): " is_paper_based
  if [[ "${is_paper_based}" == "y" ]]; then
    echo -e "## Citation\n\`\`\`bibtex\n\`\`\`" >> "${readme_file}"
  fi
}

# Main function
function main() {
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
  echo "Created experiment directory: ${NEW_EXPERIMENT_DIR}"

  # Create the .python-version file
  echo "3.12" > "${NEW_EXPERIMENT_DIR}/.python-version"
  echo "Created .python-version file: ${NEW_EXPERIMENT_DIR}/.python-version"

  # Call `deactivate` to deactivate the current virtual environment if active
  # Use command to ignore errors if deactivate doesn't exist
  command -v deactivate >/dev/null 2>&1 && deactivate || true

  # Create README.md file
  create_readme

  # Print activation instructions instead of trying to activate directly
  echo ""
  echo "To activate the virtual environment, run one of the following commands based on your shell:"
  echo ""
  echo "cd ${NEW_EXPERIMENT_DIR}"
  echo "source .venv/bin/activate.fish"
  echo "uv sync"
  echo ""
  echo "Happy hacking!"
}

# Execute main function
main "$@"
