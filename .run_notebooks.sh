# Initialize a flag to track the success of all commands
all_success=true

# Loop through all provided notebook paths
for nb in "$@"; do
    echo "Running $nb"
    # Execute the notebook and handle potential failures
    jupytext -k moscot --execute "$nb" || {
        echo "Failed to run $nb"
        all_success=false
    }
done

# Check if any executions failed
if [ "$all_success" = false ]; then
    echo "One or more notebooks failed to execute."
    exit 1
fi

echo "All notebooks executed successfully."