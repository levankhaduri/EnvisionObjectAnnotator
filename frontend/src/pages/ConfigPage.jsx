import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createSession, uploadVideo, updateConfig, extractFrames, fetchModels, listFrames } from "../api.js";

export default function ConfigPage() {
  const [sessionId, setSessionId] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [threshold, setThreshold] = useState(10);
  const [batchSize, setBatchSize] = useState(50);
  const [autoFallback, setAutoFallback] = useState(true);
  const [autoTune, setAutoTune] = useState(true);
  const [tuningTarget, setTuningTarget] = useState("0.75");
  const [tuningReserveGb, setTuningReserveGb] = useState("8");
  const [previewStride, setPreviewStride] = useState("");
  const [maxCacheFrames, setMaxCacheFrames] = useState("");
  const [maxCacheCap, setMaxCacheCap] = useState("");
  const [chunkSize, setChunkSize] = useState("");
  const [chunkSeconds, setChunkSeconds] = useState("");
  const [chunkOverlap, setChunkOverlap] = useState("1");
  const [compressMode, setCompressMode] = useState("auto");
  const [useMps, setUseMps] = useState(false);
  const [exportVideo, setExportVideo] = useState(true);
  const [exportElan, setExportElan] = useState(true);
  const [exportCsv, setExportCsv] = useState(true);
  const [outputDir, setOutputDir] = useState("");
  const [frameStride, setFrameStride] = useState(1);
  const [frameInterpolation, setFrameInterpolation] = useState("nearest");
  const [roiEnabled, setRoiEnabled] = useState(false);
  const [roiMargin, setRoiMargin] = useState(0.15);
  const [roiMinSize, setRoiMinSize] = useState(256);
  const [roiMaxCoverage, setRoiMaxCoverage] = useState(0.95);
  const [modelKey, setModelKey] = useState("auto");
  const [models, setModels] = useState([
    { key: "auto", label: "Auto (largest available)", available: true },
  ]);
  const [framesReady, setFramesReady] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [status, setStatus] = useState("Ready to start.");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
      setStatus("Loaded existing session. Upload a video or update config.");
    }
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setFramesReady(false);
      setFrameCount(0);
      return;
    }
    let mounted = true;
    listFrames(sessionId)
      .then((data) => {
        if (!mounted) return;
        const count = data.frame_count || 0;
        setFrameCount(count);
        setFramesReady(count > 0);
        if (count > 0) {
          setStatus("Frames detected. Save configuration to continue.");
        }
      })
      .catch(() => {
        if (!mounted) return;
        setFramesReady(false);
        setFrameCount(0);
      });
    return () => {
      mounted = false;
    };
  }, [sessionId]);

  useEffect(() => {
    let mounted = true;
    fetchModels()
      .then((data) => {
        if (!mounted) return;
        const apiModels = data.models || [];
        setModels([
          { key: "auto", label: "Auto (largest available)", available: true },
          ...apiModels,
        ]);
      })
      .catch(() => {
        if (!mounted) return;
      });
    return () => {
      mounted = false;
    };
  }, []);

  async function handleCreateSession() {
    try {
      setBusy(true);
      setStatus("Creating session...");
      const session = await createSession("web-session");
      setSessionId(session.id);
      localStorage.setItem("eoa_session", session.id);
      setStatus("Session created. Upload your video.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload() {
    if (!sessionId) {
      setStatus("Create a session first.");
      return;
    }
    if (!videoFile) {
      setStatus("Select a video file.");
      return;
    }
    try {
      setBusy(true);
      setStatus("Uploading video...");
      await uploadVideo(sessionId, videoFile);
      setStatus("Extracting frames...");
      await extractFrames(sessionId);
      try {
        const data = await listFrames(sessionId);
        const count = data.frame_count || 0;
        setFrameCount(count);
        setFramesReady(count > 0);
      } catch (err) {
        setFramesReady(true);
      }
      setStatus("Frames ready. Save configuration to continue.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveConfig() {
    if (!sessionId) {
      setStatus("Create a session first.");
      return;
    }
    if (!framesReady) {
      setStatus("Upload a video and extract frames before continuing.");
      return;
    }
    try {
      setBusy(true);
      setStatus("Saving configuration...");
      const toOptionalNumber = (value) => {
        if (value === "" || value === null || value === undefined) return null;
        const num = Number(value);
        return Number.isFinite(num) ? num : null;
      };
      const toOptionalInt = (value) => {
        if (value === "" || value === null || value === undefined) return null;
        const num = parseInt(value, 10);
        return Number.isFinite(num) ? num : null;
      };
      const compressMasksValue = compressMode === "auto" ? null : compressMode === "on";
      const tuningTargetValue = toOptionalNumber(tuningTarget);
      const tuningReserveValue = toOptionalNumber(tuningReserveGb);
      const roiMarginValue = toOptionalNumber(roiMargin);
      const roiMinSizeValue = toOptionalInt(roiMinSize);
      const roiMaxCoverageValue = toOptionalNumber(roiMaxCoverage);
      await updateConfig({
        session_id: sessionId,
        overlap_threshold: Number(threshold) / 100,
        batch_size: Number(batchSize),
        auto_fallback: autoFallback,
        auto_tune: autoTune,
        tuning_target: tuningTargetValue === null ? 0.75 : tuningTargetValue,
        tuning_reserve_gb: tuningReserveValue === null ? 8.0 : tuningReserveValue,
        preview_stride: toOptionalInt(previewStride),
        max_cache_frames: toOptionalInt(maxCacheFrames),
        max_cache_cap: toOptionalInt(maxCacheCap),
        use_mps: useMps,
        model_key: modelKey,
        export_video: exportVideo,
        export_elan: exportElan,
        export_csv: exportCsv,
        output_dir: outputDir || null,
        chunk_size: toOptionalInt(chunkSize),
        chunk_seconds: toOptionalNumber(chunkSeconds),
        chunk_overlap: toOptionalInt(chunkOverlap) || 1,
        compress_masks: compressMasksValue,
        frame_stride: Number(frameStride) > 1 ? Number(frameStride) : null,
        frame_interpolation: Number(frameStride) > 1 ? frameInterpolation : null,
        roi_enabled: roiEnabled,
        roi_margin: roiMarginValue === null ? 0.15 : roiMarginValue,
        roi_min_size: roiMinSizeValue === null ? 256 : roiMinSizeValue,
        roi_max_coverage: roiMaxCoverageValue === null ? 0.95 : roiMaxCoverageValue,
      });
      setStatus("Configuration saved. Proceed to annotation.");
      navigate("/annotation");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  const missingSteps = [];
  if (!sessionId) missingSteps.push("Create a session.");
  if (!framesReady) missingSteps.push("Upload a video and extract frames.");

  return (
    <div className="bg-white text-black">
      <style>{`
        body { font-family: 'Inter', sans-serif; }
        .control-panel { background: white; border: 2px solid #e5e7eb; border-radius: 12px; padding: 24px; transition: all 0.3s ease; }
        .control-panel:hover { border-color: #d1d5db; }
        .btn-primary { background: black; color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: none; cursor: pointer; }
        .btn-primary:hover:not(:disabled) { background: #374151; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .btn-primary:disabled { background: #d1d5db; cursor: not-allowed; transform: none; }
        .btn-secondary { background: white; color: black; border: 2px solid black; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; cursor: pointer; }
        .btn-secondary:hover:not(:disabled) { background: black; color: white; }
        .upload-zone { border: 3px dashed #d1d5db; border-radius: 12px; padding: 60px; text-align: center; cursor: pointer; transition: all 0.3s ease; background: white; }
        .upload-zone:hover { border-color: black; background: #f9fafb; }
        .dropdown { background: white; border: 2px solid #e5e7eb; color: black; padding: 10px 16px; border-radius: 8px; width: 100%; cursor: pointer; transition: all 0.3s ease; }
        .dropdown:focus { outline: none; border-color: black; }
        input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 8px; border-radius: 4px; background: #e5e7eb; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%; background: black; cursor: pointer; }
        input[type="range"]::-moz-range-thumb { width: 20px; height: 20px; border-radius: 50%; background: black; cursor: pointer; border: none; }
        input[type="checkbox"] { accent-color: black; }
        input[type="text"] { border: 2px solid #e5e7eb; border-radius: 8px; padding: 10px 16px; width: 100%; transition: all 0.3s ease; }
        input[type="text"]:focus { outline: none; border-color: black; }
        .video-preview { background: #000; border: 2px solid #e5e7eb; border-radius: 12px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
        video { width: 100%; height: 100%; object-fit: contain; }
        .stat-card { background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; }
        .progress-steps { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 32px; }
        .step { display: flex; align-items: center; gap: 8px; }
        .step-circle { width: 32px; height: 32px; border-radius: 50%; border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; }
        .step-circle.active { background: black; color: white; border-color: black; }
        .step-circle.completed { background: #d1d5db; color: white; border-color: #d1d5db; }
        .step-line { width: 40px; height: 2px; background: #e5e7eb; }
        .status-box { border-radius: 8px; padding: 10px 12px; font-size: 13px; margin-top: 12px; }
        .status-box.error { background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; }
      `}</style>

      <header className="bg-white border-b-2 border-black fixed w-full z-50">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-semibold flex items-center">
            <i className="fas fa-eye mr-3"></i>
            EnvisionObjectAnnotator
          </div>
          <div className="flex space-x-4">
            <Link className="btn-secondary" to="/" aria-label="Go back">
              <i className="fas fa-arrow-left mr-2"></i>Back
            </Link>
            <button className="btn-primary" onClick={handleSaveConfig} disabled={busy || missingSteps.length > 0} aria-label="Proceed to annotation">
              Next Step<i className="fas fa-arrow-right ml-2"></i>
            </button>
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
              <div className="step-circle active">2</div>
              <span className="text-sm font-medium">Configuration</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle">3</div>
              <span className="text-sm font-medium text-gray-600">Annotation</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Upload Your Video</h2>
                <div
                  className="upload-zone"
                  onClick={() => inputRef.current?.click()}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
                  />
                  <i className="fas fa-cloud-upload-alt text-4xl mb-4"></i>
                  <p className="text-lg font-medium">Drop your video here or click to upload</p>
                  <p className="text-sm text-gray-500 mt-2">Supports MP4, MOV, AVI files</p>
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <button className="btn-secondary" onClick={handleCreateSession} disabled={busy}>
                    Create Session
                  </button>
                  <button className="btn-primary" onClick={handleUpload} disabled={busy}>
                    Upload Video
                  </button>
                </div>
                <p className="text-sm text-gray-500 mt-2">Session: {sessionId || "Not created"}</p>
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Detection Settings</h2>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2">Model Selection</label>
                  <select
                    className="dropdown"
                    value={modelKey}
                    onChange={(event) => setModelKey(event.target.value)}
                  >
                    {models.map((model) => (
                      <option key={model.key} value={model.key} disabled={!model.available}>
                        {model.label}{model.available ? "" : " (missing checkpoint)"}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-2">
                    Auto selects the largest available checkpoint in your backend.
                  </p>
                </div>
                <label className="block text-sm font-medium mb-2">Overlap Threshold ({threshold}%)</label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={threshold}
                  onChange={(event) => setThreshold(event.target.value)}
                />
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-2">Batch Size</label>
                  <input
                    type="text"
                    value={batchSize}
                    onChange={(event) => setBatchSize(event.target.value)}
                  />
                </div>
                <div className="mt-4 space-y-2">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={autoFallback}
                      onChange={(event) => setAutoFallback(event.target.checked)}
                    />
                    Auto fallback to CPU if GPU memory is exhausted
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={autoTune}
                      onChange={(event) => setAutoTune(event.target.checked)}
                    />
                    Auto tune processing based on memory headroom
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={useMps}
                      onChange={(event) => setUseMps(event.target.checked)}
                    />
                    Use MPS acceleration (experimental)
                  </label>
                </div>
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Speed Controls</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Frame Stride (skip frames)</label>
                    <input
                      type="text"
                      value={frameStride}
                      onChange={(event) => setFrameStride(event.target.value)}
                    />
                    <p className="text-xs text-gray-500 mt-2">
                      Use 1 to process every frame. 2–4 gives big speedups.
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Interpolation</label>
                    <select
                      className="dropdown"
                      value={frameInterpolation}
                      onChange={(event) => setFrameInterpolation(event.target.value)}
                      disabled={Number(frameStride) <= 1}
                    >
                      <option value="nearest">Nearest (fastest)</option>
                      <option value="linear">Linear (smoother)</option>
                    </select>
                  </div>
                  <div className="border-t pt-4">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={roiEnabled}
                        onChange={(event) => setRoiEnabled(event.target.checked)}
                      />
                      Enable ROI cropping around annotated points
                    </label>
                    <div className="grid grid-cols-2 gap-3 mt-3">
                      <div>
                        <label className="block text-sm font-medium mb-2">ROI Margin (0–1)</label>
                        <input
                          type="text"
                          value={roiMargin}
                          onChange={(event) => setRoiMargin(event.target.value)}
                          disabled={!roiEnabled}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">ROI Min Size (px)</label>
                        <input
                          type="text"
                          value={roiMinSize}
                          onChange={(event) => setRoiMinSize(event.target.value)}
                          disabled={!roiEnabled}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">ROI Max Coverage (0–1)</label>
                        <input
                          type="text"
                          value={roiMaxCoverage}
                          onChange={(event) => setRoiMaxCoverage(event.target.value)}
                          disabled={!roiEnabled}
                        />
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      ROI is static; if objects move outside it, accuracy drops.
                    </p>
                  </div>
                </div>
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Advanced Processing</h2>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium mb-2">Tune Target (0–1)</label>
                    <input
                      type="text"
                      value={tuningTarget}
                      onChange={(event) => setTuningTarget(event.target.value)}
                      disabled={!autoTune}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Reserve RAM (GB)</label>
                    <input
                      type="text"
                      value={tuningReserveGb}
                      onChange={(event) => setTuningReserveGb(event.target.value)}
                      disabled={!autoTune}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Preview Stride</label>
                    <input
                      type="text"
                      value={previewStride}
                      onChange={(event) => setPreviewStride(event.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Cache Frames</label>
                    <input
                      type="text"
                      value={maxCacheFrames}
                      onChange={(event) => setMaxCacheFrames(event.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Cache Cap</label>
                    <input
                      type="text"
                      value={maxCacheCap}
                      onChange={(event) => setMaxCacheCap(event.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Chunk Size (frames)</label>
                    <input
                      type="text"
                      value={chunkSize}
                      onChange={(event) => setChunkSize(event.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Chunk Seconds</label>
                    <input
                      type="text"
                      value={chunkSeconds}
                      onChange={(event) => setChunkSeconds(event.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Chunk Overlap</label>
                    <input
                      type="text"
                      value={chunkOverlap}
                      onChange={(event) => setChunkOverlap(event.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Compress Masks</label>
                    <select
                      className="dropdown"
                      value={compressMode}
                      onChange={(event) => setCompressMode(event.target.value)}
                    >
                      <option value="auto">Auto</option>
                      <option value="on">On</option>
                      <option value="off">Off</option>
                    </select>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Leave fields blank to let auto-tune decide.
                </p>
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Output Options</h2>
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={exportVideo} onChange={(event) => setExportVideo(event.target.checked)} />
                    Save annotated video
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={exportElan} onChange={(event) => setExportElan(event.target.checked)} />
                    Export ELAN file
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={exportCsv} onChange={(event) => setExportCsv(event.target.checked)} />
                    Save CSV frame data
                  </label>
                  <div>
                    <label className="block text-sm font-medium mb-2">Output Directory (optional)</label>
                    <input type="text" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Preview</h2>
                <div className="video-preview">
                  {videoFile ? (
                    <video src={URL.createObjectURL(videoFile)} controls />
                  ) : (
                    <div className="text-gray-400">No video loaded</div>
                  )}
                </div>
                <p className="text-sm text-gray-500 mt-3">{status}</p>
                {missingSteps.length > 0 && (
                  <div className="status-box error">
                    <div className="font-semibold mb-1">Complete before continuing:</div>
                    <ul className="list-disc list-inside">
                      {missingSteps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {framesReady && frameCount > 0 && (
                  <p className="text-xs text-gray-500 mt-2">Detected {frameCount} extracted frames.</p>
                )}
              </div>

              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Quick Stats</h2>
                <div className="grid grid-cols-2 gap-3">
                  <div className="stat-card">
                    <p className="text-xs text-gray-500">Objects</p>
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
