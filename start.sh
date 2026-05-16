#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
INDEX_FILE="file://$SCRIPT_DIR/index.html"
API_DOCS="http://localhost:8000/docs"
PORT=8000

# Tarkista onko portti jo käytössä
if lsof -i ":$PORT" -sTCP:LISTEN -t &>/dev/null; then
  echo "⚠  Portti $PORT on jo käytössä. Tapetaan vanha prosessi..."
  lsof -i ":$PORT" -sTCP:LISTEN -t | xargs kill -9
  sleep 1
fi

echo "⚡ BESS-kaavoituskartoitus – käynnistetään..."
echo "   Backend:  http://localhost:$PORT"
echo "   Kartta:   $INDEX_FILE"
echo "   API docs: $API_DOCS"
echo ""

# Käynnistä backend taustalle
cd "$BACKEND_DIR"
uvicorn main:app --reload --port "$PORT" &
UVICORN_PID=$!

# Odota että backend vastaa (max 10 s)
echo "   Odotetaan backendin käynnistymistä..."
for i in $(seq 1 20); do
  if curl -s "http://localhost:$PORT/api/health" &>/dev/null; then
    echo "   Backend käynnissä ✓"
    break
  fi
  sleep 0.5
done

# Avaa selain – macOS
open "$INDEX_FILE"
sleep 0.4
open "$API_DOCS"

echo ""
echo "   Pysäytä: Ctrl+C"
echo ""

# Pidä skripti käynnissä, tapetaan backend kun skripti suljetaan
trap "echo ''; echo 'Pysäytetään backend...'; kill $UVICORN_PID 2>/dev/null; exit 0" INT TERM
wait $UVICORN_PID
