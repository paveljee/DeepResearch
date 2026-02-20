#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not configured in .env file"
    exit 1
fi

OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-"https://openrouter.ai/api/v1"}
OPENROUTER_MODEL=${OPENROUTER_MODEL:-"alibaba/tongyi-deepresearch-30b-a3b"}

#####################################
### 1. start infer               ####
#####################################

echo "==== start infer with OpenRouter... ===="

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

USE_OPENROUTER=true \
OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
OPENROUTER_BASE_URL="$OPENROUTER_BASE_URL" \
python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model "$OPENROUTER_MODEL" --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT
