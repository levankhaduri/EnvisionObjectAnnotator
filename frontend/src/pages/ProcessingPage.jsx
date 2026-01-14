import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { startProcessing, fetchProcessingStatus, getProcessingPreviewUrl } from "../api.js";

export default function ProcessingPage() {
  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
    }
  }, []);

  useEffect(() => {
    let timer;
    if (!sessionId) return;

    async function poll() {
      try {
        const data = await fetchProcessingStatus(sessionId);
        setStatus(data.status || "unknown");
        setProgress(Math.round((data.progress || 0) * 100));
        setMessage(data.message || "");
        if (data.status === "completed") {
          navigate("/results");
        }
      } catch (err) {
        setMessage(err.message);
      }
    }

    poll();
    timer = setInterval(poll, 4000);
    return () => clearInterval(timer);
  }, [sessionId, navigate]);

  useEffect(() => {
    let timer;
    if (!sessionId) return;

    const refreshPreview = () => {
      const url = `${getProcessingPreviewUrl(sessionId)}?t=${Date.now()}`;
      setPreviewUrl(url);
    };

    refreshPreview();
    timer = setInterval(refreshPreview, 3000);
    return () => clearInterval(timer);
  }, [sessionId]);

  async function handleStart() {
    if (!sessionId) {
      setMessage("No session found. Return to Config.");
      return;
    }
    try {
      setBusy(true);
      await startProcessing(sessionId);
      setMessage("Processing started.");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="bg-white text-black">
      <style>{`
        body { font-family: 'Inter', sans-serif; }
        .control-panel { background: white; border: 2px solid #e5e7eb; border-radius: 12px; padding: 24px; transition: all 0.3s ease; }
        .btn-primary { background: black; color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: none; cursor: pointer; }
        .btn-primary:hover:not(:disabled) { background: #374151; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .btn-primary:disabled { background: #d1d5db; cursor: not-allowed; }
        .btn-secondary { background: white; color: black; border: 2px solid black; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; cursor: pointer; }
        .btn-secondary:hover { background: black; color: white; }
        .progress-steps { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 32px; }
        .step { display: flex; align-items: center; gap: 8px; }
        .step-circle { width: 32px; height: 32px; border-radius: 50%; border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; }
        .step-circle.active { background: black; color: white; border-color: black; }
        .step-circle.completed { background: #d1d5db; color: white; border-color: #d1d5db; }
        .step-line { width: 40px; height: 2px; background: #e5e7eb; }
        .video-preview { background: #000; border: 2px solid #e5e7eb; border-radius: 12px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
        .progress-bar { height: 12px; background: #e5e7eb; border-radius: 6px; overflow: hidden; }
        .progress-fill { height: 100%; background: black; transition: width 0.3s ease; position: relative; }
        .progress-fill::after { content: ''; position: absolute; top: 0; left: 0; bottom: 0; right: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); animation: shimmer 2s infinite; }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
        .stat-card { background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 16px; }
        .log-entry { padding: 12px; border-left: 3px solid #e5e7eb; margin-bottom: 8px; font-size: 13px; font-family: 'Courier New', monospace; background: #f9fafb; border-radius: 4px; }
        .log-entry.info { border-left-color: #3b82f6; }
        .spinner { border: 3px solid #e5e7eb; border-top: 3px solid black; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>

      <header className="bg-white border-b-2 border-black fixed w-full z-50">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-semibold flex items-center">
            <i className="fas fa-eye mr-3"></i>
            EnvisionObjectAnnotator
          </div>
          <div className="flex space-x-4">
            <Link className="btn-secondary" to="/annotation" aria-label="Cancel processing">
              <i className="fas fa-times mr-2"></i>Cancel
            </Link>
            <Link className="btn-primary" to="/results" aria-label="View results">
              View Results<i className="fas fa-arrow-right ml-2"></i>
            </Link>
          </div>
        </nav>
      </header>

      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          <div className="progress-steps">
            <div className="step">
              <div className="step-circle completed">1</div>
              <span className="text-sm font-medium text-gray-600">Setup</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle completed">2</div>
              <span className="text-sm font-medium text-gray-600">Configuration</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle completed">3</div>
              <span className="text-sm font-medium text-gray-600">Annotation</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle active">4</div>
              <span className="text-sm font-medium">Processing</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Processing Progress</h2>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="flex justify-between text-sm text-gray-500 mt-2">
                  <span>Status: {status}</span>
                  <span>{progress}%</span>
                </div>
                <div className="mt-4">
                  <button className="btn-primary" onClick={handleStart} disabled={busy}>
                    Start Processing
                  </button>
                </div>
                <p className="text-sm text-gray-500 mt-3">{message}</p>
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Processing Log</h2>
                <div className="log-entry info">{message || "Waiting for updates..."}</div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Preview</h2>
                <div className="video-preview">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Processing preview"
                      onError={() => setPreviewUrl("")}
                      style={{ width: "100%", height: "100%", objectFit: "contain" }}
                    />
                  ) : (
                    <div className="spinner"></div>
                  )}
                </div>
                <p className="text-sm text-gray-500 mt-3">Live preview will appear here during processing.</p>
              </div>

              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Summary</h2>
                <div className="grid grid-cols-2 gap-3">
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Frames</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Objects</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Events</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Status</p>
                    <p className="text-lg font-bold">{status}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
