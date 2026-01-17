import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  API_BASE,
  listFrames,
  submitAnnotation,
  testMask,
  fetchFrameAnnotations,
} from "../api.js";

export default function AnnotationPage() {
  const [sessionId, setSessionId] = useState("");
  const [frames, setFrames] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [points, setPoints] = useState([]);
  const [objectName, setObjectName] = useState("");
  const [objectList, setObjectList] = useState([]);
  const [selectedObject, setSelectedObject] = useState("");
  const [frameObjects, setFrameObjects] = useState({});
  const [savedCounts, setSavedCounts] = useState({});
  const [videoDims, setVideoDims] = useState(null);
  const [useThumbs, setUseThumbs] = useState(true);
  const [labelMode, setLabelMode] = useState(1);
  const [status, setStatus] = useState("Load a session to begin annotating.");
  const [busy, setBusy] = useState(false);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const annotationsCache = useRef(new Map());
  const maskCache = useRef(new Map());
  const imgRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
    }
  }, []);

  useEffect(() => {
    annotationsCache.current.clear();
    maskCache.current.clear();
    setVideoDims(null);
    setUseThumbs(true);
  }, [sessionId]);

  useEffect(() => {
    async function fetchFrames() {
      if (!sessionId) return;
      try {
        const data = await listFrames(sessionId);
        setFrames(data.frame_files || []);
        setCurrentIndex(0);
        if (data.frame_width && data.frame_height) {
          setVideoDims({ width: data.frame_width, height: data.frame_height });
        }
        setUseThumbs(Boolean(data.has_thumbnails));
        setStatus(data.frame_files?.length ? "Frames ready." : "No frames found.");
      } catch (err) {
        setStatus(err.message);
      }
    }
    fetchFrames();
  }, [sessionId]);

  useEffect(() => {
    async function hydrateFrame() {
      if (!sessionId) return;
      try {
        const cacheKey = `${sessionId}:${currentIndex}`;
        const normalizeObjects = (objects) => {
          const normalized = {};
          Object.entries(objects || {}).forEach(([name, pts]) => {
            normalized[name] = (pts || []).map((p, idx) => ({
              ...p,
              fx: videoDims && videoDims.width ? p.x / videoDims.width : p.fx,
              fy: videoDims && videoDims.height ? p.y / videoDims.height : p.fy,
              id: p.id || `${currentIndex}-${name}-${idx}-${p.x}-${p.y}-${p.label}`,
            }));
          });
          return normalized;
        };

        const cached = annotationsCache.current.get(cacheKey);
        if (cached) {
          const normalized = normalizeObjects(cached);
          setFrameObjects(normalized);
          setObjectList(Object.keys(normalized));
          const counts = {};
          Object.entries(normalized).forEach(([name, pts]) => {
            counts[name] = Array.isArray(pts) ? pts.length : 0;
          });
          setSavedCounts(counts);
          const first = Object.keys(normalized)[0] || "";
          setSelectedObject(first);
          setPoints(first ? normalized[first] : []);
          return;
        }

        const data = await fetchFrameAnnotations(sessionId, currentIndex);
        const objects = data.objects || {};
        const normalized = normalizeObjects(objects);
        annotationsCache.current.set(cacheKey, normalized);
        setFrameObjects(normalized);
        setObjectList(Object.keys(normalized));
        const counts = {};
        Object.entries(normalized).forEach(([name, pts]) => {
          counts[name] = Array.isArray(pts) ? pts.length : 0;
        });
        setSavedCounts(counts);
        const first = Object.keys(normalized)[0] || "";
        setSelectedObject(first);
        setPoints(first ? normalized[first] : []);
      } catch (err) {
        setStatus(err.message);
      }
    }
    hydrateFrame();
  }, [currentIndex, sessionId, videoDims]);

  function updateCache(frameIndex, updatedObjects) {
    if (!sessionId) return;
    annotationsCache.current.set(`${sessionId}:${frameIndex}`, updatedObjects);
  }

  function removePoint(pointId) {
    if (!selectedObject) return;
    const current = frameObjects[selectedObject] || [];
    const updatedPoints = current.filter((point) => point.id !== pointId);
    const updated = { ...frameObjects, [selectedObject]: updatedPoints };
    setFrameObjects(updated);
    setPoints(updatedPoints);
    updateCache(currentIndex, updated);
    setShowPreview(false);
    setMaskPreviewUrl("");
  }

  function undoLastPoint() {
    if (!selectedObject) return;
    const current = frameObjects[selectedObject] || [];
    if (current.length === 0) return;
    const updatedPoints = current.slice(0, -1);
    const updated = { ...frameObjects, [selectedObject]: updatedPoints };
    setFrameObjects(updated);
    setPoints(updatedPoints);
    updateCache(currentIndex, updated);
    setShowPreview(false);
    setMaskPreviewUrl("");
  }

  function buildMaskSignature(pointsForObject) {
    return (pointsForObject || [])
      .map((p) => `${Math.round(p.x)}:${Math.round(p.y)}:${p.label}`)
      .join("|");
  }

  function handleImageClick(event) {
    if (!imgRef.current || !selectedObject) return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const fx = x / rect.width;
    const fy = y / rect.height;
    const baseW = videoDims?.width || imgRef.current.naturalWidth || 1;
    const baseH = videoDims?.height || imgRef.current.naturalHeight || 1;
    const nx = fx * baseW;
    const ny = fy * baseH;

    const next = [
      ...(frameObjects[selectedObject] || []),
      { x: nx, y: ny, label: labelMode, fx, fy, id: `${Date.now()}-${points.length}` },
    ];
    const updated = { ...frameObjects, [selectedObject]: next };
    setFrameObjects(updated);
    setPoints(next);
    updateCache(currentIndex, updated);
    setShowPreview(false);
    setMaskPreviewUrl("");
  }

  async function handleSavePoints() {
    if (!sessionId) {
      setStatus("Missing session. Return to Config.");
      return;
    }
    if (!selectedObject) {
      setStatus("Select or create an object before saving.");
      return;
    }
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      setStatus("Add at least one point before saving.");
      return;
    }
    try {
      setBusy(true);
      await submitAnnotation({
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: selectedObject,
        points: (frameObjects[selectedObject] || []).map(({ x, y, label }) => ({ x, y, label })),
      });
      setSavedCounts((prev) => ({
        ...prev,
        [selectedObject]: (frameObjects[selectedObject] || []).length,
      }));
      updateCache(currentIndex, frameObjects);
      setStatus("Points saved.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleTestMask() {
    if (!sessionId || !selectedObject) {
      setStatus("Select an object to test.");
      return;
    }
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      setStatus("Add at least one point before testing.");
      return;
    }
    const signature = buildMaskSignature(frameObjects[selectedObject]);
    const cacheKey = `${sessionId}:${currentIndex}:${selectedObject}:${signature}`;
    const cached = maskCache.current.get(cacheKey);
    if (cached) {
      setMaskPreviewUrl(`${API_BASE}${cached}?t=${Date.now()}`);
      setShowPreview(true);
      setStatus("Loaded cached mask preview.");
      return;
    }
    try {
      setBusy(true);
      const response = await testMask({
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: selectedObject,
        points: (frameObjects[selectedObject] || []).map(({ x, y, label }) => ({ x, y, label })),
      });
      setStatus(response.message || "Mask test complete.");
      if (response.preview_url) {
        maskCache.current.set(cacheKey, response.preview_url);
        setMaskPreviewUrl(`${API_BASE}${response.preview_url}?t=${Date.now()}`);
        setShowPreview(true);
      }
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  function handleAddObject() {
    const name = objectName.trim();
    if (!name) {
      setStatus("Enter an object name.");
      return;
    }
    if (frameObjects[name]) {
      setSelectedObject(name);
      setPoints(frameObjects[name]);
      setObjectName("");
      return;
    }
    const updated = { ...frameObjects, [name]: [] };
    setFrameObjects(updated);
    setObjectList(Object.keys(updated));
    setSavedCounts((prev) => ({ ...prev, [name]: 0 }));
    setSelectedObject(name);
    setPoints([]);
    updateCache(currentIndex, updated);
    setObjectName("");
  }

  const hasSavedObject = Object.entries(savedCounts).some(
    ([name, count]) => count > 0 && (frameObjects[name]?.length || 0) === count
  );
  const missingSteps = [];
  if (!sessionId) missingSteps.push("Load a session from the Config page.");
  if (frames.length === 0) missingSteps.push("Extract frames on the Config page.");
  if (!hasSavedObject) missingSteps.push("Save points for at least one object.");

  const frameName = frames[currentIndex];
  const frameUrl =
    sessionId && frameName
      ? `${API_BASE}${useThumbs ? "/frames/thumbs" : "/frames"}/${sessionId}/${frameName}`
      : "";

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
        .canvas-container { border: 2px solid #e5e7eb; border-radius: 12px; overflow: hidden; position: relative; background: #000; cursor: crosshair; }
        .progress-steps { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 32px; }
        .step { display: flex; align-items: center; gap: 8px; }
        .step-circle { width: 32px; height: 32px; border-radius: 50%; border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; }
        .step-circle.active { background: black; color: white; border-color: black; }
        .step-circle.completed { background: #d1d5db; color: white; border-color: #d1d5db; }
        .step-line { width: 40px; height: 2px; background: #e5e7eb; }
        .object-card { background: white; border: 2px solid #e5e7eb; border-radius: 8px; padding: 12px; transition: all 0.3s ease; }
        .object-card.active { border-color: black; background: #f9fafb; }
        .object-card:hover { border-color: #d1d5db; }
        .sticky-objects { position: sticky; top: 110px; align-self: start; }
        .point-marker { position: absolute; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; transform: translate(-50%, -50%); pointer-events: auto; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        .point-marker.positive { background: #22c55e; }
        .point-marker.negative { background: #ef4444; }
        .keyboard-hint { background: #f3f4f6; border: 2px solid #e5e7eb; border-radius: 8px; padding: 8px 12px; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
        .key { background: white; border: 2px solid #d1d5db; border-radius: 4px; padding: 2px 8px; font-weight: 600; font-size: 12px; }
        .modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.65); display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .modal-card { background: white; border-radius: 12px; padding: 16px; width: min(900px, 90vw); box-shadow: 0 20px 40px rgba(0,0,0,0.3); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .modal-close { background: black; color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
        .modal-body { background: #000; border-radius: 8px; padding: 8px; display: flex; justify-content: center; align-items: center; }
        .modal-body img { max-width: 100%; max-height: 70vh; object-fit: contain; }
        .status-box { border-radius: 8px; padding: 10px 12px; font-size: 13px; margin-top: 12px; }
        .status-box.error { background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; }
        input[type="text"] { border: 2px solid #e5e7eb; border-radius: 8px; padding: 10px 16px; width: 100%; transition: all 0.3s ease; }
        input[type="text"]:focus { outline: none; border-color: black; }
        input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 8px; border-radius: 4px; background: #e5e7eb; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%; background: black; cursor: pointer; }
      `}</style>

      <header className="bg-white border-b-2 border-black fixed w-full z-50">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="text-xl font-semibold flex items-center">
            <i className="fas fa-eye mr-3"></i>
            EnvisionObjectAnnotator
          </div>
          <div className="flex space-x-4">
            <Link className="btn-secondary" to="/config" aria-label="Go back">
              <i className="fas fa-arrow-left mr-2"></i>Back
            </Link>
            <button className="btn-primary" onClick={() => navigate("/processing")} disabled={busy || missingSteps.length > 0} aria-label="Proceed to processing">
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
              <div className="step-circle completed">2</div>
              <span className="text-sm font-medium text-gray-600">Configuration</span>
            </div>
            <div className="step-line"></div>
            <div className="step">
              <div className="step-circle active">3</div>
              <span className="text-sm font-medium">Annotation</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Annotation Canvas</h2>
                <div className="canvas-container" onClick={handleImageClick}>
                  {frameUrl ? (
                    <img
                      ref={imgRef}
                      src={frameUrl}
                      alt="Frame"
                      className="w-full h-auto"
                      onError={() => {
                        if (useThumbs) {
                          setUseThumbs(false);
                        }
                      }}
                    />
                  ) : (
                    <div className="text-gray-400 p-10">No frame loaded</div>
                  )}
                  {(frameObjects[selectedObject] || []).map((point) => (
                    <span
                      key={point.id || `${point.x}-${point.y}`}
                      className={`point-marker ${point.label === 1 ? "positive" : "negative"}`}
                      style={{ left: `${(point.fx || 0) * 100}%`, top: `${(point.fy || 0) * 100}%` }}
                      onClick={(event) => {
                        event.stopPropagation();
                        removePoint(point.id);
                      }}
                      title="Remove point"
                    />
                  ))}
                </div>
                <div className="flex items-center justify-between mt-4">
                  <button
                    className="btn-secondary"
                    onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                    disabled={currentIndex === 0}
                  >
                    Prev Frame
                  </button>
                  <span className="text-sm text-gray-500">
                    Frame {frames.length ? currentIndex + 1 : 0} / {frames.length}
                  </span>
                  <button
                    className="btn-secondary"
                    onClick={() => setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1))}
                    disabled={currentIndex >= frames.length - 1}
                  >
                    Next Frame
                  </button>
                </div>
                <div className="mt-4">
                  <label className="text-sm text-gray-500 block mb-2">Jump to frame</label>
                  <input
                    type="range"
                    min="0"
                    max={Math.max(0, frames.length - 1)}
                    value={currentIndex}
                    onChange={(event) => setCurrentIndex(Number(event.target.value))}
                    disabled={frames.length === 0}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-6 sticky-objects">
              <div className="control-panel">
                <h2 className="text-xl font-bold mb-4">Objects</h2>
                <input
                  type="text"
                  placeholder="Object name (use 'target')"
                  value={objectName}
                  onChange={(event) => setObjectName(event.target.value)}
                />
                <div className="flex gap-2 mt-3">
                  <button className="btn-secondary" onClick={handleAddObject}>Add Object</button>
                  <button className="btn-primary" onClick={handleSavePoints} disabled={busy}>Save Points</button>
                </div>
                <div className="mt-4 space-y-2">
                  {objectList.length === 0 && <div className="text-sm text-gray-500">No objects yet.</div>}
                  {objectList.map((name) => (
                    <div
                      key={name}
                      className={`object-card ${selectedObject === name ? "active" : ""}`}
                      onClick={() => {
                        setSelectedObject(name);
                        setPoints(frameObjects[name] || []);
                      }}
                    >
                      <div className="font-medium">{name}</div>
                      <div className="text-xs text-gray-500">{(frameObjects[name] || []).length} points</div>
                    </div>
                  ))}
                </div>
                {selectedObject && (frameObjects[selectedObject] || []).length > 0 && (
                  <div className="mt-4">
                    <div className="text-sm font-semibold mb-2">Points</div>
                    <div className="max-h-40 overflow-y-auto space-y-2">
                      {(frameObjects[selectedObject] || []).map((point, idx) => (
                        <div
                          key={point.id || `${point.x}-${point.y}-${idx}`}
                          className="flex items-center justify-between text-xs text-gray-600"
                        >
                          <span>
                            {idx + 1}. {point.label === 1 ? "pos" : "neg"} ({Math.round(point.x)}, {Math.round(point.y)})
                          </span>
                          <button className="btn-secondary px-2 py-1" onClick={() => removePoint(point.id)}>
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="mt-4">
                  <button className="btn-secondary" onClick={handleTestMask} disabled={busy}>Test Mask</button>
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
              </div>

              <div className="control-panel">
                <h2 className="text-2xl font-bold mb-4">Point Controls</h2>
                <div className="grid grid-cols-2 gap-4">
                  <button className="btn-primary" onClick={() => setLabelMode(1)}>
                    <i className="fas fa-plus mr-2"></i>Positive Points
                  </button>
                  <button className="btn-secondary" onClick={() => setLabelMode(0)}>
                    <i className="fas fa-minus mr-2"></i>Negative Points
                  </button>
                </div>
                <div className="mt-4">
                  <button
                    className="btn-secondary w-full"
                    onClick={undoLastPoint}
                    disabled={!selectedObject || (frameObjects[selectedObject] || []).length === 0}
                  >
                    Undo last point
                  </button>
                </div>
                <div className="flex items-center gap-4 mt-4">
                  <div className="keyboard-hint"><span className="key">C</span> Name Object</div>
                  <div className="keyboard-hint"><span className="key">T</span> Test Mask</div>
                  <div className="keyboard-hint"><span className="key">Enter</span> Confirm</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {showPreview && (
        <div className="modal-backdrop" onClick={() => setShowPreview(false)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3 className="text-lg font-semibold">Mask Preview</h3>
              <button className="modal-close" onClick={() => setShowPreview(false)}>Close</button>
            </div>
            <div className="modal-body">
              {maskPreviewUrl ? (
                <img src={maskPreviewUrl} alt="Mask preview" />
              ) : (
                <div className="text-gray-400">No preview available</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
