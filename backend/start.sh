#!/bin/bash
# BESS-työkalu backend käynnistys
# API-avain luetaan .env-tiedostosta — älä kovakoodaa avainta tähän tiedostoon

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Lataa .env jos olemassa
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "Ladattiin ympäristömuuttujat: $ENV_FILE"
fi

# Varmista että ANTHROPIC_API_KEY on asetettu
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "VAROITUS: ANTHROPIC_API_KEY ei ole asetettu — AI-strategia ei toimi"
    echo "Lisää avain tiedostoon: $ENV_FILE"
fi

cd "$SCRIPT_DIR"
exec /Users/jeremakela/Library/Python/3.9/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
