import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchResults, getResultDownloadUrl } from "../api.js";

export default function ResultsPage() {
  const [sessionId, setSessionId] = useState("");
  const [outputs, setOutputs] = useState({});
  const [status, setStatus] = useState("Loading results...");

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
    }
  }, []);

  useEffect(() => {
    async function load() {
      if (!sessionId) {
        setStatus("No session found.");
        return;
      }
      try {
        const data = await fetchResults(sessionId);
        setOutputs(data.outputs || {});
        setStatus("Results ready.");
      } catch (err) {
        setStatus(err.message);
      }
    }
    load();
  }, [sessionId]);

  return (
    <div className="bg-white text-black">
      <style>{`
        body { font-family: 'Inter', sans-serif; }
        .control-panel { background: white; border: 2px solid #e5e7eb; border-radius: 12px; padding: 24px; transition: all 0.3s ease; }
        .control-panel:hover { border-color: #d1d5db; }
        .btn-primary { background: black; color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: none; cursor: pointer; }
        .btn-primary:hover { background: #374151; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .btn-secondary { background: white; color: black; border: 2px solid black; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; cursor: pointer; }
        .btn-secondary:hover { background: black; color: white; }
        .progress-steps { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 32px; }
        .step { display: flex; align-items: center; gap: 8px; }
        .step-circle { width: 32px; height: 32px; border-radius: 50%; border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; }
        .step-circle.active { background: black; color: white; border-color: black; }
        .step-circle.completed { background: #d1d5db; color: white; border-color: #d1d5db; }
        .step-line { width: 40px; height: 2px; background: #e5e7eb; }
        .stat-card { background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 20px; text-align: center; transition: all 0.3s ease; }
        .stat-card:hover { border-color: black; transform: translateY(-2px); }
        .file-card { background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 20px; transition: all 0.3s ease; }
        .file-card:hover { border-color: #d1d5db; }
      `}</style>

      <header className="bg-white border-b-2 border-black fixed w-full z-50">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-semibold flex items-center">
            <i className="fas fa-eye mr-3"></i>
            EnvisionObjectAnnotator
          </div>
          <div className="flex space-x-4">
            <Link className="btn-secondary" to="/processing" aria-label="Back">
              <i className="fas fa-arrow-left mr-2"></i>Back
            </Link>
            <Link className="btn-primary" to="/config" aria-label="Start new">
              New Run<i className="fas fa-arrow-right ml-2"></i>
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
              <div className="step-circle completed">4</div>
              <span className="text-sm font-medium text-gray-600">Processing</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle active">5</div>
              <span className="text-sm font-medium">Results</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Download Files</h2>
                <div className="grid grid-cols-1 gap-4">
                  <div className="file-card">
                    <h3 className="font-semibold mb-2">Annotated Video</h3>
                    <a
                      className="btn-primary inline-flex"
                      href={outputs.annotated_video ? getResultDownloadUrl(sessionId, "annotated_video") : "#"}
                    >
                      Download
                    </a>
                  </div>
                  <div className="file-card">
                    <h3 className="font-semibold mb-2">CSV Output</h3>
                    <a
                      className="btn-secondary inline-flex"
                      href={outputs.csv ? getResultDownloadUrl(sessionId, "csv") : "#"}
                    >
                      Download
                    </a>
                  </div>
                  <div className="file-card">
                    <h3 className="font-semibold mb-2">ELAN Timeline</h3>
                    <a
                      className="btn-secondary inline-flex"
                      href={outputs.elan ? getResultDownloadUrl(sessionId, "elan") : "#"}
                    >
                      Download
                    </a>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-4">{status}</p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Summary</h2>
                <div className="grid grid-cols-2 gap-3">
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Events</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Targets</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Frames</p>
                    <p className="text-lg font-bold">---</p>
                  </div>
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Status</p>
                    <p className="text-lg font-bold">Ready</p>
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
