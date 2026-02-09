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

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend dev server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

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
