import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  API_BASE,
  listFrames,
  submitAnnotation,
  testMask,
  fetchFrameAnnotations,
  suggestFrames,
  deleteAnnotationObject,
} from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  TextInput,
  Checkbox,
  Slider,
  Tag,
  InlineNotification,
  InlineLoading,
  Modal,
  ProgressIndicator,
  ProgressStep,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListRow,
  StructuredListCell,
  StructuredListBody,
  DefinitionTooltip,
} from "@carbon/react";
import {
  ArrowLeft,
  ArrowRight,
  Add,
  Save,
  View,
  TrashCan,
  Undo,
  ChevronLeft,
  ChevronRight,
  FlagFilled,
  CircleFilled,
  AddAlt,
  SubtractAlt,
  Keyboard,
  Settings,
  Play,
  Image,
  WarningAlt,
  Close,
  ZoomIn,
  Cursor_1,
  TouchInteraction,
  Analytics,
  Checkmark,
} from "@carbon/icons-react";

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
  const [statusKind, setStatusKind] = useState("info");
  const [busy, setBusy] = useState(false);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const [markAsTarget, setMarkAsTarget] = useState(false);
  const [targetToggleTouched, setTargetToggleTouched] = useState(false);
  const [loading, setLoading] = useState(true);
  const [suggestedFrames, setSuggestedFrames] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestions, setSelectedSuggestions] = useState(new Set());
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const annotationsCache = useRef(new Map());
  const maskCache = useRef(new Map());
  const imgRef = useRef(null);
  const objectInputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      fetch(`${API_BASE}/sessions/${stored}`)
        .then((res) => {
          if (res.ok) {
            setSessionId(stored);
          } else {
            localStorage.removeItem("eoa_session");
            setStatus("Session expired. Please return to Config page to start a new session.");
            setStatusKind("warning");
          }
        })
        .catch(() => {
          localStorage.removeItem("eoa_session");
          setStatus("Backend unavailable. Please check if the server is running.");
          setStatusKind("error");
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      switch (e.key.toLowerCase()) {
        case "c":
          e.preventDefault();
          objectInputRef.current?.focus();
          break;
        case "t":
          e.preventDefault();
          handleTestMask();
          break;
        case "enter":
          e.preventDefault();
          handleSavePoints();
          break;
        case "z":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            undoLastPoint();
          }
          break;
        case "arrowleft":
          e.preventDefault();
          setCurrentIndex((prev) => Math.max(0, prev - 1));
          break;
        case "arrowright":
          e.preventDefault();
          setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1));
          break;
        case "1":
          e.preventDefault();
          setLabelMode(1);
          break;
        case "2":
          e.preventDefault();
          setLabelMode(0);
          break;
        default:
          break;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [frames.length, selectedObject, frameObjects, sessionId, currentIndex]);

  useEffect(() => {
    annotationsCache.current.clear();
    maskCache.current.clear();
    setVideoDims(null);
    setUseThumbs(true);
    setMarkAsTarget(false);
    setTargetToggleTouched(false);
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
        setStatusKind("info");
      } catch (err) {
        setStatus(err.message);
        setStatusKind("error");
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
          setMarkAsTarget(isTargetName(first));
          setTargetToggleTouched(false);
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
        setMarkAsTarget(isTargetName(first));
        setTargetToggleTouched(false);
      } catch (err) {
        setStatus(err.message);
        setStatusKind("error");
      }
    }
    hydrateFrame();
  }, [currentIndex, sessionId, videoDims]);

  function updateCache(frameIndex, updatedObjects) {
    if (!sessionId) return;
    annotationsCache.current.set(`${sessionId}:${frameIndex}`, updatedObjects);
  }

  function isTargetName(name) {
    return String(name || "").toLowerCase().includes("target");
  }

  function normalizeObjectName(name, targetFlag) {
    const trimmed = String(name || "").trim();
    if (!trimmed) return "";
    if (targetFlag && !isTargetName(trimmed)) {
      return `target_${trimmed}`;
    }
    return trimmed;
  }

  function displayObjectName(name) {
    const raw = String(name || "");
    const lower = raw.toLowerCase();
    if (lower.startsWith("target_")) return raw.slice("target_".length);
    if (lower.startsWith("target-")) return raw.slice("target-".length);
    if (lower.startsWith("target ")) return raw.slice("target ".length);
    return raw;
  }

  function mergePoints(primary, secondary) {
    const merged = [];
    const seen = new Set();
    const addPoint = (point) => {
      if (!point) return;
      const key = `${Math.round(point.x)}:${Math.round(point.y)}:${point.label}`;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push(point);
    };
    (primary || []).forEach(addPoint);
    (secondary || []).forEach(addPoint);
    return merged;
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

  async function handleDeleteObject(name) {
    if (!sessionId) return;
    try {
      await deleteAnnotationObject(sessionId, name);
      const updated = { ...frameObjects };
      delete updated[name];
      setFrameObjects(updated);
      setObjectList(Object.keys(updated));
      setSavedCounts((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      if (selectedObject === name) {
        const remaining = Object.keys(updated);
        setSelectedObject(remaining.length > 0 ? remaining[0] : null);
        setPoints(remaining.length > 0 ? updated[remaining[0]] || [] : []);
      }
      updateCache(currentIndex, updated);
      setStatus(`Deleted object "${name.replace("TARGET:", "")}"`);
      setStatusKind("success");
    } catch (err) {
      setStatus(err.message);
      setStatusKind("error");
    }
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
      setStatusKind("error");
      return;
    }
    if (!selectedObject) {
      setStatus("Select or create an object before saving.");
      setStatusKind("warning");
      return;
    }
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      setStatus("Add at least one point before saving.");
      setStatusKind("warning");
      return;
    }
    let objectNameToSave = selectedObject;
    let updatedObjects = frameObjects;
    let previousObjectName = null;
    if (markAsTarget) {
      const targetName = normalizeObjectName(selectedObject, true);
      if (targetName !== selectedObject) {
        updatedObjects = { ...frameObjects };
        if (updatedObjects[targetName]) {
          updatedObjects[targetName] = mergePoints(updatedObjects[targetName], updatedObjects[selectedObject]);
        } else {
          updatedObjects[targetName] = updatedObjects[selectedObject];
        }
        delete updatedObjects[selectedObject];
        objectNameToSave = targetName;
        previousObjectName = selectedObject;
      }
    } else if (isTargetName(selectedObject) && targetToggleTouched) {
      const baseName = displayObjectName(selectedObject).trim() || selectedObject;
      if (baseName !== selectedObject) {
        updatedObjects = { ...frameObjects };
        if (updatedObjects[baseName]) {
          updatedObjects[baseName] = mergePoints(updatedObjects[baseName], updatedObjects[selectedObject]);
        } else {
          updatedObjects[baseName] = updatedObjects[selectedObject];
        }
        delete updatedObjects[selectedObject];
        objectNameToSave = baseName;
        previousObjectName = selectedObject;
      }
    }
    const pointsToSave = (updatedObjects[objectNameToSave] || []).map(({ x, y, label }) => ({
      x,
      y,
      label,
    }));
    if (pointsToSave.length === 0) {
      setStatus("Add at least one point before saving.");
      setStatusKind("warning");
      return;
    }
    try {
      setBusy(true);
      await submitAnnotation({
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: objectNameToSave,
        previous_object_name: previousObjectName,
        points: pointsToSave,
      });
      setSavedCounts((prev) => ({
        ...prev,
        ...(previousObjectName ? { [previousObjectName]: 0 } : {}),
        [objectNameToSave]: pointsToSave.length,
      }));
      if (updatedObjects !== frameObjects) {
        setFrameObjects(updatedObjects);
        setObjectList(Object.keys(updatedObjects));
        setSelectedObject(objectNameToSave);
        setPoints(updatedObjects[objectNameToSave] || []);
      }
      updateCache(currentIndex, updatedObjects);
      setMarkAsTarget(isTargetName(objectNameToSave));
      setTargetToggleTouched(false);
      setStatus("Points saved successfully.");
      setStatusKind("success");
    } catch (err) {
      setStatus(err.message);
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  async function handleProcessClick() {
    // Save ALL objects across ALL annotated frames before navigating to processing
    if (!sessionId) {
      navigate("/processing");
      return;
    }
    setBusy(true);
    setStatus("Saving all annotations...");
    setStatusKind("info");
    try {
      // Collect all frames from the cache
      const framesToSave = new Map();
      for (const [key, objects] of annotationsCache.current.entries()) {
        if (!key.startsWith(`${sessionId}:`)) continue;
        const frameIdx = parseInt(key.split(":")[1], 10);
        if (isNaN(frameIdx)) continue;
        framesToSave.set(frameIdx, objects);
      }
      // Also include the current frame's live state
      if (Object.keys(frameObjects).length > 0) {
        framesToSave.set(currentIndex, frameObjects);
      }
      // Save each object in each frame
      let totalSaved = 0;
      for (const [frameIdx, objects] of framesToSave.entries()) {
        for (const [objName, objPoints] of Object.entries(objects)) {
          if (!objPoints || objPoints.length === 0) continue;
          const pointsPayload = objPoints.map(({ x, y, label }) => ({
            x,
            y,
            label: label ?? 1,
          }));
          await submitAnnotation({
            session_id: sessionId,
            frame_index: frameIdx,
            object_name: objName,
            points: pointsPayload,
          });
          totalSaved++;
        }
      }
      setStatus(`Saved ${totalSaved} object(s). Starting processing...`);
      setStatusKind("success");
      navigate("/processing");
    } catch (err) {
      setStatus(`Failed to save annotations: ${err.message}`);
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  async function handleTestMask() {
    if (!sessionId) {
      setStatus("No session. Return to Config page to start a new session.");
      setStatusKind("error");
      return;
    }
    if (!selectedObject) {
      setStatus("Select an object to test.");
      setStatusKind("warning");
      return;
    }
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      setStatus("Add at least one point before testing.");
      setStatusKind("warning");
      return;
    }
    const signature = buildMaskSignature(frameObjects[selectedObject]);
    const cacheKey = `${sessionId}:${currentIndex}:${selectedObject}:${signature}`;
    const cached = maskCache.current.get(cacheKey);
    if (cached) {
      setMaskPreviewUrl(`${API_BASE}${cached}?t=${Date.now()}`);
      setShowPreview(true);
      setStatus("Loaded cached mask preview.");
      setStatusKind("info");
      return;
    }
    try {
      setBusy(true);
      setStatus("Generating mask preview...");
      setStatusKind("info");
      const response = await testMask({
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: selectedObject,
        points: (frameObjects[selectedObject] || []).map(({ x, y, label }) => ({ x, y, label })),
      });
      setStatus(response.message || "Mask test complete.");
      setStatusKind("success");
      if (response.preview_url) {
        maskCache.current.set(cacheKey, response.preview_url);
        setMaskPreviewUrl(`${API_BASE}${response.preview_url}?t=${Date.now()}`);
        setShowPreview(true);
      }
    } catch (err) {
      console.error("Test mask error:", err);
      setStatus(`Test mask failed: ${err.message}`);
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  function handleAddObject() {
    const name = normalizeObjectName(objectName, markAsTarget);
    if (!name) {
      setStatus("Enter an object name.");
      setStatusKind("warning");
      return;
    }
    if (frameObjects[name]) {
      setSelectedObject(name);
      setPoints(frameObjects[name]);
      setObjectName("");
      setMarkAsTarget(isTargetName(name));
      setTargetToggleTouched(false);
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
    setMarkAsTarget(isTargetName(name));
    setTargetToggleTouched(false);
  }

  // --- Frame suggestion handlers ---
  async function handleSuggestFrames() {
    if (suggestedFrames.length > 0) {
      setShowSuggestions(true);
      return;
    }
    if (!sessionId || frames.length === 0) {
      setStatus("Load frames before requesting suggestions.");
      setStatusKind("warning");
      return;
    }
    setLoadingSuggestions(true);
    setStatus("Analyzing frames for optimal suggestions...");
    setStatusKind("info");
    try {
      const result = await suggestFrames(sessionId, 7, true);
      setSuggestedFrames(result.suggested_frames || []);
      setShowSuggestions(true);
      setSelectedSuggestions(new Set());
      setStatus(`Found ${result.suggested_frames?.length || 0} optimal frames (${result.method_used})`);
      setStatusKind("success");
    } catch (err) {
      setStatus(err.message);
      setStatusKind("error");
      setSuggestedFrames([]);
    } finally {
      setLoadingSuggestions(false);
    }
  }

  function toggleSuggestionSelection(frameIndex) {
    setSelectedSuggestions((prev) => {
      const updated = new Set(prev);
      if (updated.has(frameIndex)) {
        updated.delete(frameIndex);
      } else {
        updated.add(frameIndex);
      }
      return updated;
    });
  }

  function navigateToSuggestion(frameIndex) {
    setCurrentIndex(frameIndex);
    setShowSuggestions(false);
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
    <div className="app-shell">
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid #e0e0e0",
          padding: "0 1rem",
          height: "48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          backgroundColor: "#fff",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <TouchInteraction size={20} />
          <span style={{ fontWeight: 600 }}>Annotation</span>
          {selectedObject && (
            <Tag type={isTargetName(selectedObject) ? "cyan" : "gray"} size="sm">
              {displayObjectName(selectedObject)}
            </Tag>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button kind="secondary" size="sm" renderIcon={ArrowLeft} as={Link} to="/config">
            Back
          </Button>
          <Button
            size="sm"
            renderIcon={ArrowRight}
            onClick={handleProcessClick}
            disabled={busy || missingSteps.length > 0}
          >
            {busy ? "Saving..." : "Process"}
          </Button>
        </div>
      </header>

      <main className="app-content">
        <div className="page-container">
          {/* Progress indicator */}
          <div style={{ marginBottom: "1.5rem" }}>
            <ProgressIndicator currentIndex={2} spaceEqually>
              <ProgressStep label="Upload" secondaryLabel="Complete" />
              <ProgressStep label="Configure" secondaryLabel="Complete" />
              <ProgressStep label="Annotate" secondaryLabel="Mark objects" />
              <ProgressStep label="Process" secondaryLabel="Track & detect" />
            </ProgressIndicator>
          </div>

          <Grid>
            {/* Left column - Canvas */}
            <Column lg={12} md={6} sm={4}>
              <Tile style={{ padding: 0, overflow: "hidden", marginBottom: "1rem" }}>
                <div
                  style={{
                    backgroundColor: "#161616",
                    position: "relative",
                    cursor: selectedObject ? "crosshair" : "default",
                    minHeight: "300px",
                  }}
                  onClick={handleImageClick}
                >
                  {loading ? (
                    <div style={{ padding: "4rem", textAlign: "center" }}>
                      <InlineLoading description="Loading session..." />
                    </div>
                  ) : frameUrl ? (
                    <>
                      <img
                        ref={imgRef}
                        src={frameUrl}
                        alt="Frame"
                        style={{ width: "100%", height: "auto", display: "block" }}
                        onError={() => {
                          if (useThumbs) setUseThumbs(false);
                        }}
                      />
                      {!selectedObject && (
                        <div
                          style={{
                            position: "absolute",
                            inset: 0,
                            backgroundColor: "rgba(0,0,0,0.7)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <Tile style={{ maxWidth: "280px", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
                            <Cursor_1 size={32} style={{ marginBottom: "0.5rem" }} />
                            <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Create an object first</p>
                            <p style={{ fontSize: "0.875rem", color: "#525252" }}>
                              Type a name in the Objects panel and click "Add Object"
                            </p>
                          </Tile>
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ padding: "4rem", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", color: "#8d8d8d" }}>
                      {!sessionId ? (
                        <>
                          <WarningAlt size={48} style={{ marginBottom: "1rem", color: "#f1c21b" }} />
                          <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>No Active Session</p>
                          <p style={{ fontSize: "0.875rem", marginBottom: "1rem" }}>
                            Your session may have expired or the backend was restarted.
                          </p>
                          <Button kind="secondary" size="sm" as={Link} to="/config" renderIcon={ArrowLeft}>
                            Go to Config
                          </Button>
                        </>
                      ) : (
                        <>
                          <Image size={48} style={{ marginBottom: "1rem", opacity: 0.5 }} />
                          <p>No frames loaded</p>
                        </>
                      )}
                    </div>
                  )}
                  {/* Point markers */}
                  {(frameObjects[selectedObject] || []).map((point) => (
                    <span
                      key={point.id || `${point.x}-${point.y}`}
                      style={{
                        position: "absolute",
                        left: `${(point.fx || 0) * 100}%`,
                        top: `${(point.fy || 0) * 100}%`,
                        width: "18px",
                        height: "18px",
                        borderRadius: "50%",
                        border: "3px solid white",
                        transform: "translate(-50%, -50%)",
                        backgroundColor: point.label === 1 ? "#42be65" : "#da1e28",
                        cursor: "pointer",
                        boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
                        transition: "transform 0.15s ease",
                      }}
                      onClick={(event) => {
                        event.stopPropagation();
                        removePoint(point.id);
                      }}
                      title="Click to remove"
                    />
                  ))}
                </div>
              </Tile>

              {/* Frame navigation */}
              <Tile>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={ChevronLeft}
                    onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                    disabled={currentIndex === 0}
                    iconDescription="Previous frame"
                  >
                    Prev
                  </Button>
                  <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                    Frame {frames.length ? currentIndex + 1 : 0} / {frames.length}
                  </span>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={ChevronRight}
                    onClick={() => setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1))}
                    disabled={currentIndex >= frames.length - 1}
                    iconDescription="Next frame"
                  >
                    Next
                  </Button>
                </div>
                <Slider
                  id="frame-slider"
                  labelText="Jump to frame"
                  min={0}
                  max={Math.max(0, frames.length - 1)}
                  value={currentIndex}
                  onChange={({ value }) => setCurrentIndex(value)}
                  disabled={frames.length === 0}
                  hideTextInput
                />
              </Tile>
            </Column>

            {/* Right column - Controls */}
            <Column lg={4} md={2} sm={4}>
              <div style={{ maxHeight: "calc(100vh - 120px)", overflowY: "auto", position: "sticky", top: "60px" }}>
              {/* Objects panel */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <View size={16} /> Objects
                </h3>
                <TextInput
                  ref={objectInputRef}
                  id="object-name"
                  labelText="Object name"
                  placeholder="e.g., hand, cup, ball"
                  value={objectName}
                  onChange={(e) => setObjectName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddObject();
                    }
                  }}
                  size="sm"
                />
                <Checkbox
                  id="mark-target"
                  labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><FlagFilled size={14} /> Mark as <DefinitionTooltip definition="The object you want to detect gaze/overlap against. At least one target is needed for ELAN export and overlap analysis." align="bottom">target object</DefinitionTooltip></span>}
                  checked={markAsTarget}
                  onChange={(_, { checked }) => {
                    setMarkAsTarget(checked);
                    setTargetToggleTouched(true);
                  }}
                  style={{ marginTop: "0.75rem" }}
                />
                <p style={{ fontSize: "0.75rem", color: "#6f6f6f", marginTop: "0.25rem", marginLeft: "1.75rem" }}>
                  Required for gaze/overlap detection and ELAN export
                </p>
                <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
                  <Button kind="secondary" size="sm" onClick={handleAddObject} renderIcon={Add}>
                    Add
                  </Button>
                  <Button size="sm" onClick={handleSavePoints} disabled={busy} renderIcon={Save}>
                    Save
                  </Button>
                </div>

                {/* Object list */}
                <div style={{ marginTop: "1rem" }}>
                  {objectList.length === 0 ? (
                    <p style={{ fontSize: "0.875rem", color: "#6f6f6f" }}>No objects yet.</p>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                      {objectList.map((name) => (
                        <div
                          key={name}
                          onClick={() => {
                            setSelectedObject(name);
                            setPoints(frameObjects[name] || []);
                            setMarkAsTarget(isTargetName(name));
                            setTargetToggleTouched(false);
                          }}
                          style={{
                            padding: "0.75rem",
                            borderRadius: "4px",
                            border: `2px solid ${selectedObject === name ? "#0f62fe" : "#e0e0e0"}`,
                            backgroundColor: selectedObject === name ? "#e5f6ff" : "#fff",
                            cursor: "pointer",
                            transition: "all 0.15s ease",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <span style={{ fontWeight: 500, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                              {displayObjectName(name)}
                              {isTargetName(name) && <Tag type="cyan" size="sm">Target</Tag>}
                            </span>
                            <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                              <Tag type="gray" size="sm">{(frameObjects[name] || []).length} pts</Tag>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteObject(name);
                                }}
                                title="Delete object"
                                style={{
                                  background: "none",
                                  border: "none",
                                  cursor: "pointer",
                                  padding: "2px",
                                  display: "flex",
                                  alignItems: "center",
                                  color: "#da1e28",
                                  opacity: 0.6,
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                                onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.6)}
                              >
                                <Close size={16} />
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {objectList.length > 0 && !objectList.some((n) => isTargetName(n)) && (
                  <InlineNotification
                    kind="info"
                    lowContrast
                    hideCloseButton
                    subtitle="Mark at least one object as a target to enable gaze detection and ELAN export."
                    style={{ marginTop: "0.75rem", minBlockSize: "auto" }}
                  />
                )}
              </Tile>

              {/* Point mode */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <CircleFilled size={16} /> Point Mode
                </h3>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <Button
                    kind={labelMode === 1 ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => setLabelMode(1)}
                    renderIcon={AddAlt}
                    style={{ flex: 1 }}
                  >
                    Positive
                  </Button>
                  <Button
                    kind={labelMode === 0 ? "danger" : "secondary"}
                    size="sm"
                    onClick={() => setLabelMode(0)}
                    renderIcon={SubtractAlt}
                    style={{ flex: 1 }}
                  >
                    Negative
                  </Button>
                </div>
                <p style={{ fontSize: "0.75rem", color: "#525252", marginTop: "0.5rem" }}>
                  {labelMode === 1 ? "Click to mark object regions (green)" : "Click to exclude regions (red)"}
                </p>
                <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
                  <Button
                    kind="ghost"
                    size="sm"
                    onClick={undoLastPoint}
                    disabled={!selectedObject || (frameObjects[selectedObject] || []).length === 0}
                    renderIcon={Undo}
                    style={{ flex: 1 }}
                  >
                    Undo
                  </Button>
                  <Button
                    kind="tertiary"
                    size="sm"
                    onClick={handleTestMask}
                    disabled={busy}
                    renderIcon={ZoomIn}
                    style={{ flex: 1 }}
                  >
                    Test Mask
                  </Button>
                </div>
              </Tile>

              {/* Suggest Frames */}
              <Tile style={{ marginBottom: "1rem" }}>
                {loadingSuggestions ? (
                  <InlineLoading
                    description="Analyzing frames for optimal suggestions..."
                  />
                ) : (
                  <Button
                    kind="tertiary"
                    size="sm"
                    renderIcon={Analytics}
                    onClick={handleSuggestFrames}
                    disabled={frames.length === 0}
                    style={{ width: "100%" }}
                  >
                    {suggestedFrames.length > 0
                      ? `View Suggested Frames (${suggestedFrames.length})`
                      : "Suggest Optimal Frames"}
                  </Button>
                )}
              </Tile>

              {/* Keyboard shortcuts */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Keyboard size={16} /> Shortcuts
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem", fontSize: "0.75rem" }}>
                  <div><Tag type="gray" size="sm">C</Tag> Focus name</div>
                  <div><Tag type="gray" size="sm">T</Tag> Test mask</div>
                  <div><Tag type="gray" size="sm">Enter</Tag> Save</div>
                  <div><Tag type="gray" size="sm">Ctrl+Z</Tag> Undo</div>
                  <div><Tag type="gray" size="sm">1</Tag> Positive</div>
                  <div><Tag type="gray" size="sm">2</Tag> Negative</div>
                  <div><Tag type="gray" size="sm">←→</Tag> Frames</div>
                </div>
              </Tile>

              {/* Status */}
              {status && (
                <InlineNotification
                  kind={statusKind}
                  title={status}
                  lowContrast
                  hideCloseButton
                  style={{ marginBottom: "1rem" }}
                />
              )}

              {/* Missing steps warning */}
              {missingSteps.length > 0 && (
                <InlineNotification
                  kind="warning"
                  title="Complete before continuing"
                  lowContrast
                  hideCloseButton
                >
                  <ul style={{ margin: "0.5rem 0 0 1rem", padding: 0 }}>
                    {missingSteps.map((step) => (
                      <li key={step} style={{ fontSize: "0.875rem" }}>{step}</li>
                    ))}
                  </ul>
                </InlineNotification>
              )}
              </div>
            </Column>
          </Grid>
        </div>
      </main>

      {/* Mask preview modal */}
      <Modal
        open={showPreview}
        onRequestClose={() => setShowPreview(false)}
        modalHeading="Mask Preview"
        passiveModal
        size="lg"
      >
        <div style={{ backgroundColor: "#000", borderRadius: "4px", padding: "1rem", display: "flex", justifyContent: "center" }}>
          {maskPreviewUrl ? (
            <img src={maskPreviewUrl} alt="Mask preview" style={{ maxWidth: "100%", maxHeight: "60vh", objectFit: "contain" }} />
          ) : (
            <p style={{ color: "#6f6f6f" }}>No preview available</p>
          )}
        </div>
      </Modal>

      {/* Frame suggestion modal */}
      <Modal
        open={showSuggestions}
        onRequestClose={() => setShowSuggestions(false)}
        modalHeading="Suggested Optimal Frames"
        passiveModal
        size="lg"
      >
        <p style={{ fontSize: "0.875rem", color: "#525252", marginBottom: "1rem" }}>
          Select frame(s) to annotate. Higher scores indicate better quality and content.
        </p>
        {suggestedFrames.length > 0 ? (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "1rem" }}>
              {suggestedFrames.map((frame) => {
                const isSelected = selectedSuggestions.has(frame.frame_index);
                const frameName = frames[frame.frame_index];
                const thumbUrl = frameName
                  ? `${API_BASE}/frames/thumbs/${sessionId}/${frameName}`
                  : "";

                return (
                  <label
                    key={frame.frame_index}
                    htmlFor={`suggest-check-${frame.frame_index}`}
                    style={{
                      border: `2px solid ${isSelected ? "#0f62fe" : "#e0e0e0"}`,
                      borderRadius: "4px",
                      overflow: "hidden",
                      cursor: "pointer",
                      backgroundColor: isSelected ? "#e5f6ff" : "#fff",
                      transition: "all 0.15s ease",
                      display: "block",
                    }}
                  >
                    <div style={{ padding: "0.75rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <Checkbox
                        id={`suggest-check-${frame.frame_index}`}
                        checked={isSelected}
                        onChange={() => toggleSuggestionSelection(frame.frame_index)}
                        labelText={`Frame ${frame.frame_index + 1}`}
                        style={{ minWidth: 0 }}
                      />
                      <Tag type={frame.method === "dinov2" ? "green" : "blue"} size="sm">
                        {(frame.score * 100).toFixed(0)}%
                      </Tag>
                    </div>
                    {thumbUrl && (
                      <img
                        src={thumbUrl}
                        alt={`Frame ${frame.frame_index}`}
                        style={{ width: "100%", height: "120px", objectFit: "cover" }}
                      />
                    )}
                    <div style={{ padding: "0.5rem 0.75rem", fontSize: "0.75rem" }}>
                      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                        <span style={{ color: "#525252" }}>Sharpness: <strong>{frame.sharpness.toFixed(1)}</strong></span>
                        <span style={{ color: "#525252" }}>Brightness: <strong>{(frame.brightness * 100).toFixed(0)}%</strong></span>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "1.5rem", paddingTop: "1rem", borderTop: "1px solid #e0e0e0" }}>
              <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                {selectedSuggestions.size > 0
                  ? `${selectedSuggestions.size} frame(s) selected`
                  : "Click frames to select"}
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <Button
                  kind="secondary"
                  size="sm"
                  onClick={() => {
                    if (selectedSuggestions.size > 0) {
                      navigateToSuggestion(Math.min(...Array.from(selectedSuggestions)));
                    }
                  }}
                  disabled={selectedSuggestions.size === 0}
                >
                  Go to Selected
                </Button>
                <Button
                  kind="primary"
                  size="sm"
                  renderIcon={Checkmark}
                  onClick={() => {
                    if (selectedSuggestions.size > 0) {
                      const indices = Array.from(selectedSuggestions).sort((a, b) => a - b);
                      setCurrentIndex(indices[0]);
                      setShowSuggestions(false);
                      setStatus(`Ready to annotate ${indices.length} selected frame(s).`);
                      setStatusKind("success");
                    }
                  }}
                  disabled={selectedSuggestions.size === 0}
                >
                  Start Annotating ({selectedSuggestions.size})
                </Button>
              </div>
            </div>
          </>
        ) : (
          <p style={{ color: "#6f6f6f", textAlign: "center", padding: "2rem 0" }}>
            No frame suggestions available.
          </p>
        )}
      </Modal>
    </div>
  );
}
