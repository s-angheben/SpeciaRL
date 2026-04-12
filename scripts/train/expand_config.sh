#!/bin/bash
# Script to expand environment variables in YAML configuration using pure bash
# Usage: ./expand_config.sh input.yml output.yml

set -e

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 input.yml output.yml"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Expand ${VAR} references in a YAML file. For api_base lines, only the
# currently loaded model (LOADED_MODEL_PATH) gets its real api_base; all
# others are rewritten to a disabled placeholder.
expand_variables() {
    local content
    content=$(cat "$1")

    local current_model_path=""

    while IFS= read -r line; do
        if [[ "$line" =~ model_path:[[:space:]]*[\"\']?([^\"\']+)[\"\']?[[:space:]]*$ ]]; then
            current_model_path="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^[[:space:]]*-[[:space:]]*name: ]]; then
            current_model_path=""
        fi

        if [[ "$line" =~ api_base: ]]; then
            if [[ -n "$LOADED_MODEL_PATH" && "$current_model_path" == "$LOADED_MODEL_PATH" ]]; then
                while [[ "$line" =~ \$\{([A-Za-z_][A-Za-z0-9_]*)\} ]]; do
                    var_name="${BASH_REMATCH[1]}"
                    var_value="${!var_name}"
                    line="${line/\$\{$var_name\}/$var_value}"
                done
                echo "$line"
            else
                echo "    api_base: \"http://disabled-model:9999/\""
            fi
        else
            while [[ "$line" =~ \$\{([A-Za-z_][A-Za-z0-9_]*)\} ]]; do
                var_name="${BASH_REMATCH[1]}"
                var_value="${!var_name}"
                line="${line/\$\{$var_name\}/$var_value}"
            done
            echo "$line"
        fi
    done <<< "$content"
}

# Expand variables and write to output file
expand_variables "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Configuration expanded from '$INPUT_FILE' to '$OUTPUT_FILE'"