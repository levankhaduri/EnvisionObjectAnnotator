import React, { useEffect, useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { API_BASE, startProcessing, fetchProcessingStatus, getProcessingPreviewUrl } from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  Tag,
  InlineNotification,
  InlineLoading,
  ProgressBar,
  ProgressIndicator,
  ProgressStep,
} from "@carbon/react";
import {
  ArrowLeft,
  ArrowRight,
  Hourglass,
  Checkmark,
  ErrorFilled,
  Time,
  Laptop,
  DataBase,
  Dashboard,
  Activity,
  Video,
  CircleFilled,
  WarningAlt,
  ChartLine,
} from "@carbon/icons-react";

export default function ProcessingPage() {
  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [systemStats, setSystemStats] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [statusError, setStatusError] = useState(null);
  const [fetchErrorCount, setFetchErrorCount] = useState(0);
  const hasStarted = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
    }
  }, []);

  // Auto-start processing when session is loaded
  useEffect(() => {
    if (!sessionId || hasStarted.current) return;

    async function autoStart() {
      try {
        hasStarted.current = true;
        await startProcessing(sessionId);
        setStartTime(Date.now());
        setMessage("Processing started...");
      } catch (err) {
        if (!err.message.includes("already")) {
          setMessage(err.message);
        }
        setStartTime(Date.now());
      }
    }

    autoStart();
  }, [sessionId]);

  // Poll processing status
  useEffect(() => {
    let timer;
    let errorCount = 0;
    let lastStatus = "idle";

    if (!sessionId) return;

    async function poll() {
      try {
        const data = await fetchProcessingStatus(sessionId);
        setStatusError(null);
        setFetchErrorCount(0);
        errorCount = 0;
        lastStatus = data.status || "unknown";
        setStatus(lastStatus);
        setProgress(Math.round((data.progress || 0) * 100));
        setMessage(data.message || "");
        if (data.status === "completed") {
          navigate("/results");
        }
      } catch (err) {
        errorCount++;
        setFetchErrorCount(errorCount);
        setStatusError(err.message);
        if (errorCount >= 3 && (lastStatus === "processing" || lastStatus === "saving")) {
          setMessage("Connection lost - processing may have completed. Check results.");
        }
      }
    }

    poll();
    timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, [sessionId, navigate]);

  // Poll system stats
  useEffect(() => {
    let timer;

    async function fetchStats() {
      try {
        const res = await fetch(`${API_BASE}/system/stats`);
        if (res.ok) {
          const data = await res.json();
          setSystemStats(data);
        }
      } catch (err) {
        // Silent fail
      }
    }

    fetchStats();
    timer = setInterval(fetchStats, 2000);
    return () => clearInterval(timer);
  }, []);

  // Update elapsed time
  useEffect(() => {
    let timer;
    if (startTime && status !== "completed" && status !== "error") {
      timer = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [startTime, status]);

  // Refresh preview
  useEffect(() => {
    let timer;
    if (!sessionId) return;

    const refreshPreview = () => {
      const url = `${getProcessingPreviewUrl(sessionId)}?t=${Date.now()}`;
      setPreviewUrl(url);
    };

    refreshPreview();
    timer = setInterval(refreshPreview, 4000);
    return () => clearInterval(timer);
  }, [sessionId]);

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function getStatusTag() {
    switch (status) {
      case "processing":
        return <Tag type="blue" size="sm"><CircleFilled size={10} style={{ marginRight: 4, animation: "pulse 1.5s infinite" }} /> Processing</Tag>;
      case "completed":
        return <Tag type="green" size="sm"><Checkmark size={12} /> Complete</Tag>;
      case "error":
        return <Tag type="red" size="sm"><ErrorFilled size={12} /> Error</Tag>;
      case "saving":
        return <Tag type="purple" size="sm"><Hourglass size={12} /> Saving</Tag>;
      case "initializing":
        return <Tag type="teal" size="sm"><Hourglass size={12} /> Loading Model</Tag>;
      default:
        return <Tag type="gray" size="sm">Initializing</Tag>;
    }
  }

  function StatRing({ value, color, label, sublabel }) {
    const radius = 34;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div style={{ position: "relative", width: "80px", height: "80px" }}>
          <svg width="80" height="80" style={{ transform: "rotate(-90deg)" }}>
            <circle
              cx="40"
              cy="40"
              r={radius}
              fill="none"
              stroke="#e0e0e0"
              strokeWidth="6"
            />
            <circle
              cx="40"
              cy="40"
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth="6"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.5s ease" }}
            />
          </svg>
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 600,
            fontSize: "1rem"
          }}>
            {value}%
          </div>
        </div>
        <div>
          <div style={{ fontWeight: 500, fontSize: "0.875rem" }}>{label}</div>
          <div style={{ color: "#6f6f6f", fontSize: "0.75rem" }}>{sublabel}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* Add pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

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
          <Activity size={20} />
          <span style={{ fontWeight: 600 }}>Processing</span>
          {getStatusTag()}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button kind="secondary" size="sm" renderIcon={ArrowLeft} as={Link} to="/annotation">
            Back
          </Button>
          <Button size="sm" renderIcon={ArrowRight} as={Link} to="/results">
            Results
          </Button>
        </div>
      </header>

      <main className="app-content">
        <div className="page-container">
          {/* Progress indicator */}
          <div style={{ marginBottom: "1.5rem" }}>
            <ProgressIndicator currentIndex={3} spaceEqually>
              <ProgressStep label="Upload" secondaryLabel="Complete" />
              <ProgressStep label="Configure" secondaryLabel="Complete" />
              <ProgressStep label="Annotate" secondaryLabel="Complete" />
              <ProgressStep label="Process" secondaryLabel="Running..." />
            </ProgressIndicator>
          </div>

          {/* Main progress card */}
          <Tile style={{ marginBottom: "1.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <div style={{
                  width: "48px",
                  height: "48px",
                  borderRadius: "50%",
                  backgroundColor: status === "processing" ? "#e5f6ff" : status === "completed" ? "#defbe6" : "#f4f4f4",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  {status === "processing" ? (
                    <InlineLoading style={{ margin: 0 }} />
                  ) : status === "completed" ? (
                    <Checkmark size={24} style={{ color: "#198038" }} />
                  ) : (
                    <Hourglass size={24} style={{ color: "#525252" }} />
                  )}
                </div>
                <div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{progress}%</div>
                  <div style={{ fontSize: "0.875rem", color: "#525252" }}>{message || "Processing..."}</div>
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#525252" }}>
                  <Time size={16} />
                  <span style={{ fontSize: "1.25rem", fontWeight: 500 }}>{formatTime(elapsedTime)}</span>
                </div>
                <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>Elapsed time</div>
              </div>
            </div>

            <ProgressBar
              value={progress}
              size="big"
              status={status === "error" ? "error" : status === "completed" ? "finished" : "active"}
            />

            {statusError && fetchErrorCount >= 2 && (
              <InlineNotification
                kind="warning"
                title="Connection issue"
                subtitle={statusError}
                lowContrast
                hideCloseButton
                style={{ marginTop: "1rem" }}
                actions={
                  <Button kind="ghost" size="sm" as={Link} to="/results">
                    Check Results
                  </Button>
                }
              />
            )}
          </Tile>

          <Grid>
            {/* Preview */}
            <Column lg={12} md={6} sm={4}>
              <Tile style={{ padding: 0, overflow: "hidden" }}>
                <div style={{
                  backgroundColor: "#161616",
                  aspectRatio: "16/9",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Processing preview"
                      onError={() => setPreviewUrl("")}
                      style={{ width: "100%", height: "100%", objectFit: "contain" }}
                    />
                  ) : (
                    <div style={{ textAlign: "center", color: "#8d8d8d" }}>
                      <InlineLoading description="Waiting for preview..." />
                    </div>
                  )}
                </div>
                <div style={{ padding: "0.75rem", backgroundColor: "#262626", color: "#c6c6c6", fontSize: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Video size={14} /> Live preview updates every 4 seconds
                </div>
              </Tile>
            </Column>

            {/* System Stats */}
            <Column lg={4} md={2} sm={4}>
              {/* GPU */}
              {systemStats?.gpu && (
                <Tile style={{ marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Laptop size={16} />
                      <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>GPU</span>
                    </div>
                    <Tag type="cool-gray" size="sm">{systemStats.gpu.name}</Tag>
                  </div>
                  <StatRing
                    value={systemStats.gpu.used_pct}
                    color="#0f62fe"
                    label={`${systemStats.gpu.reserved_gb} GB used`}
                    sublabel={`of ${systemStats.gpu.total_gb} GB`}
                  />
                </Tile>
              )}

              {/* RAM */}
              {systemStats?.ram && (
                <Tile style={{ marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <DataBase size={16} />
                      <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>Memory</span>
                    </div>
                    <Tag type="cool-gray" size="sm">RAM</Tag>
                  </div>
                  <StatRing
                    value={systemStats.ram.used_pct}
                    color="#42be65"
                    label={`${systemStats.ram.used_gb} GB used`}
                    sublabel={`${systemStats.ram.available_gb} GB available`}
                  />
                </Tile>
              )}

              {/* CPU */}
              {systemStats?.cpu && (
                <Tile style={{ marginBottom: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Dashboard size={16} />
                      <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>CPU</span>
                    </div>
                    <Tag type="cool-gray" size="sm">{systemStats.cpu.cores} cores</Tag>
                  </div>
                  <StatRing
                    value={systemStats.cpu.percent}
                    color="#f1c21b"
                    label={`${systemStats.cpu.percent}% usage`}
                    sublabel={`${systemStats.cpu.cores} cores`}
                  />
                </Tile>
              )}

              {/* Fallback if no stats */}
              {!systemStats?.gpu && !systemStats?.ram && !systemStats?.cpu && (
                <Tile style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", padding: "2rem" }}>
                  <Activity size={32} style={{ marginBottom: "0.5rem", color: "#6f6f6f" }} />
                  <p style={{ fontSize: "0.875rem", color: "#6f6f6f" }}>Loading system stats...</p>
                </Tile>
              )}
            </Column>
          </Grid>
        </div>
      </main>
    </div>
  );
}
