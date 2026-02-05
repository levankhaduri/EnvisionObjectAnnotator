import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  createSession,
  uploadVideo,
  updateConfig,
  extractFrames,
  fetchModels,
  listFrames,
  createSampleClip,
  detectGreyStart,
} from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  TextInput,
  NumberInput,
  Select,
  SelectItem,
  Checkbox,
  Toggle,
  Slider,
  Accordion,
  AccordionItem,
  InlineNotification,
  ProgressIndicator,
  ProgressStep,
  FileUploaderDropContainer,
  Tag,
  Link as CarbonLink,
  InlineLoading,
} from "@carbon/react";
import {
  ArrowLeft,
  ArrowRight,
  Upload,
  Checkmark,
  Video,
  Settings,
  Play,
  Document,
  Home,
  Cut,
  Timer,
  Meter,
  MachineLearning,
  SettingsAdjust,
  Export,
  DocumentExport,
  Time,
  Crop,
  Rocket,
  Folder,
  VideoFilled,
  Number_1,
  Number_2,
  Number_3,
  CheckmarkFilled,
  ChartMultitype,
  Filter,
} from "@carbon/icons-react";

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
  const [processStartFrame, setProcessStartFrame] = useState("");
  const [processEndFrame, setProcessEndFrame] = useState("");
  const [sampleDuration, setSampleDuration] = useState("10");
  const [trimStart, setTrimStart] = useState("");
  const [trimEnd, setTrimEnd] = useState("");
  const videoRef = useRef(null);
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
  const [speedPreset, setSpeedPreset] = useState("slow");
  const [modelKey, setModelKey] = useState("auto");
  const [models, setModels] = useState([
    { key: "auto", label: "Auto (largest available)", available: true },
  ]);
  const [framesReady, setFramesReady] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [videoUploaded, setVideoUploaded] = useState(false);
  const [status, setStatus] = useState("Ready to start.");
  const [statusType, setStatusType] = useState("info");
  const [busy, setBusy] = useState(false);
  const [sessionStale, setSessionStale] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  // Speed presets effect
  useEffect(() => {
    if (speedPreset === "custom") return;
    const presets = {
      slow: { frameStride: 1, roiEnabled: false },
      medium: { frameStride: 2, roiEnabled: true },
      fast: { frameStride: 4, roiEnabled: true },
      ultra: { frameStride: 6, roiEnabled: true },
    };
    const preset = presets[speedPreset];
    if (!preset) return;
    setFrameStride(preset.frameStride);
    setRoiEnabled(preset.roiEnabled);
  }, [speedPreset]);

  // Session validation on mount
  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      fetch(`${import.meta.env.VITE_API_BASE || "http://localhost:8000"}/sessions/${stored}`)
        .then((res) => {
          if (res.ok) {
            return res.json().then((sessionData) => {
              setSessionId(stored);
              if (sessionData.video_path) {
                setVideoUploaded(true);
                setStatus("Session restored. Checking frames...");
              }
            });
          } else {
            localStorage.removeItem("eoa_session");
            setSessionStale(true);
            setStatus("Previous session expired. Upload a video to start fresh.");
            setStatusType("warning");
          }
        })
        .catch(() => {
          localStorage.removeItem("eoa_session");
          setSessionStale(true);
          setStatus("Cannot connect to backend. Make sure the server is running.");
          setStatusType("error");
        });
    }
  }, []);

  // Check frames when session changes
  useEffect(() => {
    if (!sessionId) {
      setFramesReady(false);
      setFrameCount(0);
      return;
    }
    listFrames(sessionId)
      .then((data) => {
        const count = data.frame_count || 0;
        setFrameCount(count);
        setFramesReady(count > 0);
        if (count > 0) {
          setStatus(`${count} frames detected. Ready to continue.`);
          setStatusType("success");
        }
      })
      .catch(() => {
        setFramesReady(false);
        setFrameCount(0);
      });
  }, [sessionId]);

  // Fetch models
  useEffect(() => {
    fetchModels()
      .then((data) => {
        const apiModels = data.models || [];
        setModels([
          { key: "auto", label: "Auto (largest available)", available: true },
          ...apiModels,
        ]);
      })
      .catch(() => {});
  }, []);

  async function handleFileUpload(files) {
    const file = files[0];
    if (!file) return;
    setVideoFile(file);
    try {
      setBusy(true);
      setStatus("Creating session...");
      setStatusType("info");
      const session = await createSession("web-session");
      setSessionId(session.id);
      localStorage.setItem("eoa_session", session.id);
      setVideoUploaded(false);
      setFramesReady(false);
      setFrameCount(0);
      setStatus("Uploading video...");
      await uploadVideo(session.id, file);
      setVideoUploaded(true);
      setSessionStale(false);
      setStatus("Video uploaded. Extract frames to continue.");
      setStatusType("success");
    } catch (err) {
      setStatus(err.message);
      setStatusType("error");
    } finally {
      setBusy(false);
    }
  }

  async function handleExtractFrames() {
    if (!sessionId || !videoUploaded) return;
    try {
      setBusy(true);
      const trimInfo = trimStart || trimEnd ? ` (${trimStart || "0"}s - ${trimEnd || "end"})` : "";
      setStatus(`Extracting frames${trimInfo}...`);
      setStatusType("info");
      await extractFrames(sessionId, 2, trimStart || null, trimEnd || null);
      const data = await listFrames(sessionId);
      const count = data.frame_count || 0;
      setFrameCount(count);
      setFramesReady(count > 0);
      setStatus(`${count} frames extracted. Ready to continue.`);
      setStatusType("success");
    } catch (err) {
      setStatus(err.message);
      setStatusType("error");
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveConfig() {
    if (!sessionId || !framesReady) return;
    try {
      setBusy(true);
      setStatus("Saving configuration...");
      setStatusType("info");
      const toNum = (v) => (v === "" ? null : Number(v) || null);
      const toInt = (v) => (v === "" ? null : parseInt(v, 10) || null);
      await updateConfig({
        session_id: sessionId,
        overlap_threshold: Number(threshold) / 100,
        batch_size: Number(batchSize),
        auto_fallback: autoFallback,
        auto_tune: autoTune,
        tuning_target: toNum(tuningTarget) || 0.75,
        tuning_reserve_gb: toNum(tuningReserveGb) || 8.0,
        preview_stride: toInt(previewStride),
        max_cache_frames: toInt(maxCacheFrames),
        use_mps: useMps,
        model_key: modelKey,
        export_video: exportVideo,
        export_elan: exportElan,
        export_csv: exportCsv,
        output_dir: outputDir || null,
        chunk_size: toInt(chunkSize),
        chunk_seconds: toNum(chunkSeconds),
        chunk_overlap: toInt(chunkOverlap) || 1,
        compress_masks: compressMode === "auto" ? null : compressMode === "on",
        frame_stride: frameStride > 1 ? frameStride : null,
        frame_interpolation: frameStride > 1 ? frameInterpolation : null,
        roi_enabled: roiEnabled,
        roi_margin: roiMargin,
        roi_min_size: roiMinSize,
        roi_max_coverage: roiMaxCoverage,
        process_start_frame: toInt(processStartFrame),
        process_end_frame: toInt(processEndFrame),
      });
      setStatus("Configuration saved!");
      setStatusType("success");
      navigate("/annotation");
    } catch (err) {
      setStatus(err.message);
      setStatusType("error");
    } finally {
      setBusy(false);
    }
  }

  const currentStep = framesReady ? 2 : videoUploaded ? 1 : 0;

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
          <Settings size={20} />
          <span style={{ fontWeight: 600 }}>Configuration</span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button kind="secondary" size="sm" renderIcon={ArrowLeft} as={Link} to="/">
            Back
          </Button>
          <Button
            size="sm"
            renderIcon={ArrowRight}
            onClick={handleSaveConfig}
            disabled={busy || !framesReady}
          >
            Next Step
          </Button>
        </div>
      </header>

      {/* Main content */}
      <main className="app-content">
        <div className="page-container">
          {/* Progress indicator */}
          <div style={{ marginBottom: "2rem" }}>
            <ProgressIndicator currentIndex={currentStep} spaceEqually>
              <ProgressStep label="Upload" secondaryLabel="Select video" />
              <ProgressStep label="Extract" secondaryLabel="Process frames" />
              <ProgressStep label="Configure" secondaryLabel="Set options" />
            </ProgressIndicator>
          </div>

          <Grid>
            {/* Left column - Steps */}
            <Column lg={4} md={4} sm={4}>
              {/* Step 1: Upload */}
              <Tile style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                  {videoUploaded ? (
                    <Tag type="green" size="sm"><CheckmarkFilled size={12} /> Uploaded</Tag>
                  ) : (
                    <Tag type="gray" size="sm"><Number_1 size={12} /></Tag>
                  )}
                  <Upload size={16} style={{ color: videoUploaded ? "#198038" : "#525252" }} />
                  <span style={{ fontWeight: 600 }}>Upload Video</span>
                </div>

                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: "none" }}
                  onChange={(e) => handleFileUpload(e.target.files)}
                />

                <Button
                  kind={videoUploaded ? "tertiary" : "primary"}
                  size="sm"
                  style={{ width: "100%" }}
                  onClick={() => inputRef.current?.click()}
                  disabled={busy}
                  renderIcon={Upload}
                >
                  {busy && !videoUploaded ? "Uploading..." : videoUploaded ? "Change Video" : "Select Video"}
                </Button>

                {videoFile && (
                  <p style={{ fontSize: "0.75rem", color: "#525252", marginTop: "0.5rem" }}>
                    {videoFile.name}
                  </p>
                )}
              </Tile>

              {/* Step 2: Extract */}
              <Tile style={{ marginBottom: "1rem", opacity: videoUploaded ? 1 : 0.5 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                  {framesReady ? (
                    <Tag type="green" size="sm"><CheckmarkFilled size={12} /> {frameCount} frames</Tag>
                  ) : (
                    <Tag type={videoUploaded ? "blue" : "gray"} size="sm"><Number_2 size={12} /></Tag>
                  )}
                  <Cut size={16} style={{ color: framesReady ? "#198038" : "#525252" }} />
                  <span style={{ fontWeight: 600 }}>Extract Frames</span>
                </div>

                <Button
                  kind={framesReady ? "tertiary" : "primary"}
                  size="sm"
                  style={{ width: "100%" }}
                  onClick={handleExtractFrames}
                  disabled={busy || !videoUploaded}
                  renderIcon={Video}
                >
                  {busy && videoUploaded && !framesReady ? "Extracting..." : framesReady ? "Re-extract" : "Extract Frames"}
                </Button>
              </Tile>

              {/* Step 3: Continue */}
              <Tile style={{ opacity: framesReady ? 1 : 0.5 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                  <Tag type={framesReady ? "blue" : "gray"} size="sm"><Number_3 size={12} /></Tag>
                  <Play size={16} style={{ color: framesReady ? "#0f62fe" : "#525252" }} />
                  <span style={{ fontWeight: 600 }}>Start Annotation</span>
                </div>

                <Button
                  size="sm"
                  style={{ width: "100%" }}
                  onClick={handleSaveConfig}
                  disabled={busy || !framesReady}
                  renderIcon={Play}
                >
                  Continue
                </Button>
              </Tile>

              {/* Status notification */}
              {status && (
                <div style={{ marginTop: "1rem" }}>
                  <InlineNotification
                    kind={statusType === "error" ? "error" : statusType === "warning" ? "warning" : statusType === "success" ? "success" : "info"}
                    title={status}
                    lowContrast
                    hideCloseButton
                  />
                </div>
              )}
            </Column>

            {/* Center column - Video preview */}
            <Column lg={8} md={4} sm={4}>
              <Tile style={{ padding: 0, overflow: "hidden" }}>
                <div
                  style={{
                    backgroundColor: "#000",
                    aspectRatio: "16/9",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {videoFile ? (
                    <video
                      ref={videoRef}
                      src={URL.createObjectURL(videoFile)}
                      controls
                      style={{ width: "100%", height: "100%", objectFit: "contain" }}
                    />
                  ) : (
                    <div style={{ textAlign: "center", color: "#8d8d8d", padding: "2rem" }}>
                      <div style={{
                        width: "120px",
                        height: "120px",
                        borderRadius: "50%",
                        background: "linear-gradient(135deg, #393939 0%, #262626 100%)",
                        margin: "0 auto 1.5rem",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "3px dashed #525252"
                      }}>
                        <VideoFilled size={48} style={{ color: "#6f6f6f" }} />
                      </div>
                      <p style={{ fontSize: "1rem", fontWeight: 500, marginBottom: "0.5rem" }}>
                        No video selected
                      </p>
                      <p style={{ fontSize: "0.875rem", opacity: 0.7 }}>
                        Upload a video file to begin
                      </p>
                      <div style={{
                        marginTop: "1.5rem",
                        display: "flex",
                        justifyContent: "center",
                        gap: "0.75rem",
                        flexWrap: "wrap"
                      }}>
                        <Tag type="cool-gray" size="sm">MP4</Tag>
                        <Tag type="cool-gray" size="sm">MOV</Tag>
                        <Tag type="cool-gray" size="sm">AVI</Tag>
                        <Tag type="cool-gray" size="sm">MKV</Tag>
                      </div>
                    </div>
                  )}
                </div>
              </Tile>

              {/* Trim controls */}
              {videoUploaded && !framesReady && (
                <Tile style={{ marginTop: "1rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                    <span style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Cut size={16} /> Trim Video (optional)
                    </span>
                    <Button
                      kind="ghost"
                      size="sm"
                      onClick={async () => {
                        try {
                          setStatus("Scanning for grey frames...");
                          const result = await detectGreyStart(sessionId);
                          if (result.first_valid_time > 0) {
                            setTrimStart(String(result.first_valid_time));
                            setStatus(`Found ${result.first_valid_frame} grey frames. Start set to ${result.first_valid_time}s`);
                            setStatusType("success");
                          } else {
                            setStatus("No grey frames detected");
                            setStatusType("info");
                          }
                        } catch (err) {
                          setStatus(err.message);
                          setStatusType("error");
                        }
                      }}
                      disabled={busy}
                    >
                      Auto-detect grey start
                    </Button>
                  </div>
                  <Grid narrow>
                    <Column lg={8} md={4} sm={2}>
                      <TextInput
                        id="trim-start"
                        labelText="Start (seconds)"
                        placeholder="0"
                        value={trimStart}
                        onChange={(e) => setTrimStart(e.target.value)}
                        size="sm"
                      />
                    </Column>
                    <Column lg={8} md={4} sm={2}>
                      <TextInput
                        id="trim-end"
                        labelText="End (seconds)"
                        placeholder="end"
                        value={trimEnd}
                        onChange={(e) => setTrimEnd(e.target.value)}
                        size="sm"
                      />
                    </Column>
                  </Grid>
                </Tile>
              )}

              {/* Ready indicator */}
              {framesReady && (
                <InlineNotification
                  kind="success"
                  title={`${frameCount} frames ready`}
                  subtitle="Click 'Continue' to start annotation"
                  lowContrast
                  hideCloseButton
                  style={{ marginTop: "1rem" }}
                />
              )}
            </Column>

            {/* Right column - Settings */}
            <Column lg={4} md={8} sm={4}>
              <Accordion>
                {/* Model & Detection */}
                <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><MachineLearning size={16} /> Model & Detection</span>}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <Select
                      id="model-select"
                      labelText="Model"
                      value={modelKey}
                      onChange={(e) => setModelKey(e.target.value)}
                    >
                      {models.map((model) => (
                        <SelectItem
                          key={model.key}
                          value={model.key}
                          text={model.label + (model.available ? "" : " (missing)")}
                          disabled={!model.available}
                        />
                      ))}
                    </Select>

                    <div>
                      <label style={{ fontSize: "0.75rem", color: "#525252" }}>
                        Overlap Threshold: {threshold}%
                      </label>
                      <Slider
                        min={1}
                        max={50}
                        value={threshold}
                        onChange={({ value }) => setThreshold(value)}
                        hideTextInput
                      />
                    </div>

                    <TextInput
                      id="batch-size"
                      labelText="Batch Size"
                      value={String(batchSize)}
                      onChange={(e) => setBatchSize(e.target.value)}
                      size="sm"
                    />

                    <Checkbox
                      id="auto-fallback"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Filter size={14} /> Auto fallback to CPU</span>}
                      checked={autoFallback}
                      onChange={(_, { checked }) => setAutoFallback(checked)}
                    />
                    <Checkbox
                      id="auto-tune"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Meter size={14} /> Auto tune for memory</span>}
                      checked={autoTune}
                      onChange={(_, { checked }) => setAutoTune(checked)}
                    />
                  </div>
                </AccordionItem>

                {/* Speed Controls */}
                <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Rocket size={16} /> Speed Controls</span>}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <Select
                      id="speed-preset"
                      labelText="Performance Preset"
                      value={speedPreset}
                      onChange={(e) => setSpeedPreset(e.target.value)}
                    >
                      <SelectItem value="slow" text="Slow (best quality)" />
                      <SelectItem value="medium" text="Medium" />
                      <SelectItem value="fast" text="Fast" />
                      <SelectItem value="ultra" text="Ultra" />
                      <SelectItem value="custom" text="Custom" />
                    </Select>

                    {speedPreset !== "slow" && (
                      <InlineNotification
                        kind="warning"
                        title="Faster = lower quality"
                        lowContrast
                        hideCloseButton
                      />
                    )}

                    <TextInput
                      id="frame-stride"
                      labelText="Frame Stride"
                      value={String(frameStride)}
                      onChange={(e) => {
                        setFrameStride(Number(e.target.value) || 1);
                        setSpeedPreset("custom");
                      }}
                      size="sm"
                    />

                    <Checkbox
                      id="roi-enabled"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Crop size={14} /> Enable ROI crop</span>}
                      checked={roiEnabled}
                      onChange={(_, { checked }) => {
                        setRoiEnabled(checked);
                        setSpeedPreset("custom");
                      }}
                    />
                  </div>
                </AccordionItem>

                {/* Output Options */}
                <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Export size={16} /> Output Options</span>}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <Checkbox
                      id="export-video"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Video size={14} /> Annotated video</span>}
                      checked={exportVideo}
                      onChange={(_, { checked }) => setExportVideo(checked)}
                    />
                    <Checkbox
                      id="export-elan"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><ChartMultitype size={14} /> ELAN file (.eaf)</span>}
                      checked={exportElan}
                      onChange={(_, { checked }) => setExportElan(checked)}
                    />
                    <Checkbox
                      id="export-csv"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><DocumentExport size={14} /> CSV data</span>}
                      checked={exportCsv}
                      onChange={(_, { checked }) => setExportCsv(checked)}
                    />
                    <TextInput
                      id="output-dir"
                      labelText={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Folder size={14} /> Output Directory</span>}
                      placeholder="(optional)"
                      value={outputDir}
                      onChange={(e) => setOutputDir(e.target.value)}
                      size="sm"
                    />
                  </div>
                </AccordionItem>

                {/* Advanced */}
                <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><SettingsAdjust size={16} /> Advanced (rarely needed)</span>}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                    <TextInput
                      id="tune-target"
                      labelText="Tune target"
                      value={tuningTarget}
                      onChange={(e) => setTuningTarget(e.target.value)}
                      size="sm"
                      disabled={!autoTune}
                    />
                    <TextInput
                      id="reserve-gb"
                      labelText="Reserve GB"
                      value={tuningReserveGb}
                      onChange={(e) => setTuningReserveGb(e.target.value)}
                      size="sm"
                      disabled={!autoTune}
                    />
                    <TextInput
                      id="preview-stride"
                      labelText="Preview stride"
                      value={previewStride}
                      onChange={(e) => setPreviewStride(e.target.value)}
                      size="sm"
                    />
                    <TextInput
                      id="max-cache"
                      labelText="Max cache"
                      value={maxCacheFrames}
                      onChange={(e) => setMaxCacheFrames(e.target.value)}
                      size="sm"
                    />
                    <TextInput
                      id="chunk-size"
                      labelText="Chunk size"
                      value={chunkSize}
                      onChange={(e) => setChunkSize(e.target.value)}
                      size="sm"
                    />
                    <TextInput
                      id="start-frame"
                      labelText="Start frame"
                      value={processStartFrame}
                      onChange={(e) => setProcessStartFrame(e.target.value)}
                      size="sm"
                    />
                    <TextInput
                      id="end-frame"
                      labelText="End frame"
                      value={processEndFrame}
                      onChange={(e) => setProcessEndFrame(e.target.value)}
                      size="sm"
                    />
                    <Select
                      id="compress"
                      labelText="Compress"
                      value={compressMode}
                      onChange={(e) => setCompressMode(e.target.value)}
                      size="sm"
                    >
                      <SelectItem value="auto" text="Auto" />
                      <SelectItem value="on" text="On" />
                      <SelectItem value="off" text="Off" />
                    </Select>
                  </div>
                </AccordionItem>
              </Accordion>
            </Column>
          </Grid>
        </div>
      </main>
    </div>
  );
}
