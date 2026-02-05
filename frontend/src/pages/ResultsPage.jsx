import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchResults, getResultDownloadUrl } from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  ClickableTile,
  Tag,
  InlineNotification,
  InlineLoading,
  Accordion,
  AccordionItem,
  ProgressIndicator,
  ProgressStep,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListRow,
  StructuredListCell,
  StructuredListBody,
} from "@carbon/react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  Video,
  Document,
  ChartMultitype,
  Checkmark,
  Time,
  Laptop,
  DataBase,
  Activity,
  Star,
  Calendar,
  Reset,
  Table,
  CheckmarkFilled,
  WarningAlt,
  Hourglass,
} from "@carbon/icons-react";

export default function ResultsPage() {
  const [sessionId, setSessionId] = useState("");
  const [outputs, setOutputs] = useState({});
  const [outputsMeta, setOutputsMeta] = useState({});
  const [profiling, setProfiling] = useState(null);
  const [status, setStatus] = useState("Loading results...");
  const [statusKind, setStatusKind] = useState("info");

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      setSessionId(stored);
    }
  }, []);

  const csvStatus = outputsMeta?.csv_status;
  const csvProgress = outputsMeta?.csv_progress;
  const csvError = outputsMeta?.csv_error;
  const csvPending = !outputs.csv && (csvStatus === "pending" || csvStatus === "running");

  useEffect(() => {
    let timer;
    async function load() {
      if (!sessionId) {
        setStatus("No session found.");
        setStatusKind("warning");
        return;
      }
      try {
        const data = await fetchResults(sessionId);
        setOutputs(data.outputs || {});
        setOutputsMeta(data.outputs_meta || {});
        setProfiling(data.profiling || null);
        setStatus("Results ready.");
        setStatusKind("success");
      } catch (err) {
        setStatus(err.message);
        setStatusKind("error");
      }
    }
    load();
    if (sessionId && csvPending) {
      timer = setInterval(load, 5000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [sessionId, csvPending]);

  const formatSeconds = (value) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "---";
    return `${value.toFixed(2)}s`;
  };
  const formatFloat = (value, decimals = 2) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "---";
    return value.toFixed(decimals);
  };

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
          <Star size={20} />
          <span style={{ fontWeight: 600 }}>Results</span>
          <Tag type="green" size="sm"><CheckmarkFilled size={12} /> Complete</Tag>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button kind="secondary" size="sm" renderIcon={ArrowLeft} as={Link} to="/processing">
            Back
          </Button>
          <Button size="sm" renderIcon={Reset} as={Link} to="/config">
            New Run
          </Button>
        </div>
      </header>

      <main className="app-content">
        <div className="page-container">
          {/* Progress indicator */}
          <div style={{ marginBottom: "1.5rem" }}>
            <ProgressIndicator currentIndex={4} spaceEqually>
              <ProgressStep label="Upload" secondaryLabel="Complete" />
              <ProgressStep label="Configure" secondaryLabel="Complete" />
              <ProgressStep label="Annotate" secondaryLabel="Complete" />
              <ProgressStep label="Process" secondaryLabel="Complete" />
              <ProgressStep label="Results" secondaryLabel="Download" />
            </ProgressIndicator>
          </div>

          {/* Success banner */}
          <Tile style={{
            marginBottom: "1.5rem",
            backgroundColor: "#defbe6",
            border: "1px solid #a7f0ba"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
              <div style={{
                width: "48px",
                height: "48px",
                borderRadius: "50%",
                backgroundColor: "#198038",
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                <Checkmark size={24} style={{ color: "#fff" }} />
              </div>
              <div>
                <div style={{ fontSize: "1.25rem", fontWeight: 600, color: "#198038" }}>
                  Processing Complete
                </div>
                <div style={{ fontSize: "0.875rem", color: "#525252" }}>
                  Your files are ready for download
                </div>
              </div>
            </div>
          </Tile>

          <Grid>
            {/* Download section */}
            <Column lg={10} md={5} sm={4}>
              <Tile style={{ marginBottom: "1rem" }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Download size={20} /> Download Files
                </h2>

                <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                  {/* Annotated Video */}
                  <div style={{
                    padding: "1rem",
                    borderRadius: "4px",
                    border: "1px solid #e0e0e0",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                      <div style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "8px",
                        backgroundColor: "#e5f6ff",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center"
                      }}>
                        <Video size={20} style={{ color: "#0f62fe" }} />
                      </div>
                      <div>
                        <div style={{ fontWeight: 600 }}>Annotated Video</div>
                        <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>MP4 with mask overlays</div>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      kind="primary"
                      renderIcon={Download}
                      href={outputs.annotated_video ? getResultDownloadUrl(sessionId, "annotated_video") : "#"}
                      disabled={!outputs.annotated_video}
                    >
                      Download
                    </Button>
                  </div>

                  {/* CSV */}
                  <div style={{
                    padding: "1rem",
                    borderRadius: "4px",
                    border: "1px solid #e0e0e0",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                      <div style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "8px",
                        backgroundColor: "#defbe6",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center"
                      }}>
                        <Table size={20} style={{ color: "#198038" }} />
                      </div>
                      <div>
                        <div style={{ fontWeight: 600 }}>CSV Data</div>
                        <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>
                          {csvPending ? (
                            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                              <Hourglass size={12} />
                              Exporting{typeof csvProgress === "number" ? ` ${Math.round(csvProgress * 100)}%` : "..."}
                            </span>
                          ) : csvStatus === "disabled" ? (
                            "Export disabled"
                          ) : csvStatus === "error" ? (
                            <span style={{ color: "#da1e28" }}>Export failed{csvError ? `: ${csvError}` : ""}</span>
                          ) : (
                            "Frame-by-frame data"
                          )}
                        </div>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      kind="secondary"
                      renderIcon={Download}
                      href={outputs.csv ? getResultDownloadUrl(sessionId, "csv") : "#"}
                      disabled={!outputs.csv}
                    >
                      Download
                    </Button>
                  </div>

                  {/* ELAN */}
                  <div style={{
                    padding: "1rem",
                    borderRadius: "4px",
                    border: "1px solid #e0e0e0",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                      <div style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "8px",
                        backgroundColor: "#fff1f1",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center"
                      }}>
                        <ChartMultitype size={20} style={{ color: "#da1e28" }} />
                      </div>
                      <div>
                        <div style={{ fontWeight: 600 }}>ELAN Timeline</div>
                        <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>EAF annotation file</div>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      kind="secondary"
                      renderIcon={Download}
                      href={outputs.elan ? getResultDownloadUrl(sessionId, "elan") : "#"}
                      disabled={!outputs.elan}
                    >
                      Download
                    </Button>
                  </div>
                </div>

                {/* Status notification */}
                <InlineNotification
                  kind={statusKind}
                  title={status}
                  lowContrast
                  hideCloseButton
                  style={{ marginTop: "1rem" }}
                />
              </Tile>
            </Column>

            {/* Stats and profiling */}
            <Column lg={6} md={3} sm={4}>
              {/* Summary stats */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Activity size={16} /> Summary
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                  <div style={{ padding: "0.75rem", backgroundColor: "#f4f4f4", borderRadius: "4px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>Events</div>
                    <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{profiling?.objects_total ?? "---"}</div>
                  </div>
                  <div style={{ padding: "0.75rem", backgroundColor: "#f4f4f4", borderRadius: "4px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>Frames</div>
                    <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{profiling?.frames_total ?? "---"}</div>
                  </div>
                  <div style={{ padding: "0.75rem", backgroundColor: "#f4f4f4", borderRadius: "4px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>FPS</div>
                    <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{formatFloat(profiling?.processing_fps)}</div>
                  </div>
                  <div style={{ padding: "0.75rem", backgroundColor: "#f4f4f4", borderRadius: "4px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#6f6f6f" }}>Total Time</div>
                    <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{formatSeconds(profiling?.timings_s?.total)}</div>
                  </div>
                </div>
              </Tile>

              {/* Profiling details */}
              {profiling && (
                <Accordion>
                  <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Time size={16} /> Timing Details</span>}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.875rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Load annotations</span>
                        <span>{formatSeconds(profiling.timings_s?.load_annotations)}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Model init</span>
                        <span>{formatSeconds(profiling.timings_s?.model_init)}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Processing</span>
                        <span>{formatSeconds(profiling.timings_s?.processing)}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Saving outputs</span>
                        <span>{formatSeconds(profiling.timings_s?.saving)}</span>
                      </div>
                    </div>
                  </AccordionItem>

                  <AccordionItem title={<span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}><Laptop size={16} /> Hardware</span>}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.875rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Model</span>
                        <span>{profiling.model_label || profiling.model_key || "---"}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "#6f6f6f" }}>Device</span>
                        <span>{profiling.device || "---"}</span>
                      </div>
                      {profiling.gpu_mem_end && (
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#6f6f6f" }}>GPU Memory</span>
                          <span>{formatFloat(profiling.gpu_mem_end.allocated_gb)} / {formatFloat(profiling.gpu_mem_end.reserved_gb)} GB</span>
                        </div>
                      )}
                      {profiling.ram_end && (
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#6f6f6f" }}>RAM</span>
                          <span>{formatFloat(profiling.ram_end.used_gb)} GB used</span>
                        </div>
                      )}
                    </div>
                  </AccordionItem>
                </Accordion>
              )}
            </Column>
          </Grid>
        </div>
      </main>
    </div>
  );
}
