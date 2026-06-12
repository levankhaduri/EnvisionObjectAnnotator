#!/bin/bash
# EnvisionObjectAnnotator Run Script (Mac/Linux)
# Run: ./run.sh

echo ""
echo "Starting EnvisionObjectAnnotator..."
echo ""

# Check if setup was done
if [ ! -d "backend/.venv" ]; then
    echo "ERROR: Backend not set up. Run ./setup.sh first."
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "ERROR: Frontend not set up. Run ./setup.sh first."
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend API server..."
cd backend
./.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Poll /health until the backend responds. Torch + SAM2 import regularly
# takes longer than the old hardcoded 3s wait.
echo "  Waiting for backend to be ready (up to 60s)..."
backend_ready=0
for i in $(seq 1 60); do
    if curl -sf --max-time 1 http://localhost:8000/health > /dev/null 2>&1; then
        backend_ready=1
        echo "  Backend ready after ${i}s"
        break
    fi
    sleep 1
done
if [ "$backend_ready" -ne 1 ]; then
    echo "  WARNING: Backend did not respond within 60s. Check stderr above."
fi

# Start frontend
echo "Starting frontend dev server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Poll the Vite dev server until it serves something.
echo "  Waiting for frontend to be ready (up to 30s)..."
frontend_ready=0
for i in $(seq 1 30); do
    if curl -sf --max-time 1 http://localhost:5173 > /dev/null 2>&1; then
        frontend_ready=1
        echo "  Frontend ready after ${i}s"
        break
    fi
    sleep 1
done
if [ "$frontend_ready" -ne 1 ]; then
    echo "  WARNING: Frontend did not respond within 30s."
fi

echo ""
echo "========================================"
echo "  EnvisionObjectAnnotator Running!"
echo "========================================"
echo ""
echo "Open in browser: http://localhost:5173"
echo ""
echo "Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Try to open browser
if [[ "$(uname)" == "Darwin" ]]; then
    open "http://localhost:5173" 2>/dev/null || true
else
    xdg-open "http://localhost:5173" 2>/dev/null || true
fi

# Wait for processes
wait
