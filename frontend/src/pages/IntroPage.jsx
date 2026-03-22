import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { runDiagnostics, API_BASE } from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  InlineNotification,
  InlineLoading,
  Accordion,
  AccordionItem,
  Tag,
  Link as CarbonLink,
} from "@carbon/react";
import {
  Play,
  Checkmark,
  Close,
  LogoGithub,
  View,
  Video,
  DocumentExport,
  Analytics,
} from "@carbon/icons-react";

export default function IntroPage() {
  const navigate = useNavigate();
  const [diagResults, setDiagResults] = useState(null);
  const [diagLoading, setDiagLoading] = useState(false);
  const [diagError, setDiagError] = useState(null);

  async function handleRunDiagnostics() {
    setDiagLoading(true);
    setDiagError(null);
    try {
      const results = await runDiagnostics();
      setDiagResults(results);
    } catch (err) {
      setDiagError(err.message);
    } finally {
      setDiagLoading(false);
    }
  }

  function DiagnosticItem({ ok, label, value, children }) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "0.75rem",
          padding: "0.75rem",
          backgroundColor: ok ? "#defbe6" : "#fff1f1",
          borderRadius: "4px",
          marginBottom: "0.5rem",
        }}
      >
        {ok ? (
          <Checkmark size={20} style={{ color: "#198038", flexShrink: 0 }} />
        ) : (
          <Close size={20} style={{ color: "#da1e28", flexShrink: 0 }} />
        )}
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontWeight: 600 }}>{label}</span>
            {value && (
              <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                {value}
              </span>
            )}
          </div>
          {children}
        </div>
      </div>
    );
  }

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
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "4px",
              backgroundColor: "#0f62fe",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <View size={20} style={{ color: "#fff" }} />
          </div>
          <span style={{ fontWeight: 600, fontSize: "1rem" }}>
            EnvisionObjectAnnotator
          </span>
        </div>
        <CarbonLink
          href="https://github.com/DavAhm/EnvisionObjectAnnotator"
          target="_blank"
          rel="noreferrer"
          style={{ display: "flex", alignItems: "center" }}
        >
          <LogoGithub size={24} />
        </CarbonLink>
      </header>

      {/* Main content */}
      <main className="app-content">
        <div className="page-container">
          <Grid>
            {/* Hero section */}
            <Column lg={16} md={8} sm={4}>
              <div style={{ textAlign: "center", padding: "3rem 0 2.5rem" }}>
                <h1
                  style={{
                    fontSize: "2.625rem",
                    fontWeight: 300,
                    marginBottom: "1rem",
                    lineHeight: 1.2,
                  }}
                >
                  Video Object Annotation
                </h1>
                <p
                  style={{
                    fontSize: "1.25rem",
                    color: "#525252",
                    maxWidth: "640px",
                    margin: "0 auto 2rem",
                  }}
                >
                  Annotate objects across multiple frames, track them bidirectionally with SAM2, and detect overlaps between designated objects (e.g., gaze tracker and object; or hand and object).
                </p>
                <Button
                  size="lg"
                  renderIcon={Play}
                  onClick={() => navigate("/config")}
                >
                  Start Annotating
                </Button>
              </div>
            </Column>

            {/* Feature cards */}
            {[
              { icon: View, title: "Point-based annotation", desc: "Click to mark objects on one or more reference frames" },
              { icon: Video, title: "SAM2 tracking", desc: "Forward and backward propagation with multiple model sizes" },
              { icon: Analytics, title: "Overlap detection", desc: "Automatic target-to-object overlap tracking with ELAN and CSV export" },
              { icon: DocumentExport, title: "Smart frame suggestion", desc: "Automatically find the sharpest, most informative frames to annotate" },
            ].map(({ icon: Icon, title, desc }) => (
              <Column key={title} lg={4} md={4} sm={4}>
                <Tile style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  textAlign: "center",
                  padding: "2rem 1.5rem",
                  height: "100%",
                }}>
                  <Icon size={32} style={{ marginBottom: "1rem", color: "#0f62fe" }} />
                  <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                    {title}
                  </h3>
                  <p style={{ fontSize: "0.875rem", color: "#525252" }}>
                    {desc}
                  </p>
                </Tile>
              </Column>
            ))}

            {/* Setup & Diagnostics accordion */}
            <Column lg={{ span: 12, offset: 2 }} md={8} sm={4} style={{ marginTop: "2rem" }}>
              <Accordion>
                <AccordionItem title="First time? Setup Instructions">
                  <div style={{ padding: "0.5rem 0" }}>
                    <p style={{ marginBottom: "1rem", color: "#525252" }}>
                      <strong>Prerequisites:</strong> Python 3.10+, Node.js 18+, Git, ffmpeg
                    </p>
                    <pre
                      style={{
                        backgroundColor: "#161616",
                        color: "#f4f4f4",
                        padding: "1rem",
                        borderRadius: "4px",
                        fontSize: "0.875rem",
                        overflow: "auto",
                      }}
                    >
{`# Clone and setup
git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git
cd EnvisionObjectAnnotator

# Windows
.\\setup.ps1
.\\run.ps1

# Mac/Linux
./setup.sh
./run.sh`}
                    </pre>
                    <p style={{ fontSize: "0.75rem", color: "#6f6f6f", marginTop: "1rem" }}>
                      Setup downloads SAM2, EdgeTAM, and model checkpoints automatically.
                    </p>
                    <div style={{ marginTop: "1rem" }}>
                      <CarbonLink
                        href="https://github.com/DavAhm/EnvisionObjectAnnotator#manual-setup-advanced"
                        target="_blank"
                        rel="noreferrer"
                      >
                        View detailed manual setup instructions
                      </CarbonLink>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem title="System Diagnostics">
                  <div style={{ padding: "0.5rem 0" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: "1rem",
                      }}
                    >
                      <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                        Verify your setup is working correctly
                      </span>
                      <Button
                        size="sm"
                        kind="tertiary"
                        onClick={handleRunDiagnostics}
                        disabled={diagLoading}
                      >
                        {diagLoading ? (
                          <InlineLoading description="Checking..." />
                        ) : (
                          "Run Diagnostics"
                        )}
                      </Button>
                    </div>

                    {diagError && (
                      <InlineNotification
                        kind="error"
                        title="Connection failed"
                        subtitle={`Make sure the backend is running at ${API_BASE}`}
                        lowContrast
                        style={{ marginBottom: "1rem" }}
                      />
                    )}

                    {diagResults && (
                      <div>
                        <DiagnosticItem
                          ok={diagResults.python?.ok}
                          label="Python"
                          value={diagResults.python?.version?.split(" ")[0]}
                        />

                        <DiagnosticItem
                          ok={diagResults.pytorch?.ok}
                          label="PyTorch"
                          value={diagResults.pytorch?.version}
                        >
                          {diagResults.pytorch?.ok && (
                            <div style={{ fontSize: "0.75rem", marginTop: "0.25rem" }}>
                              {diagResults.pytorch?.cuda_available ? (
                                <span style={{ color: "#198038" }}>
                                  CUDA {diagResults.pytorch?.cuda_version} ·{" "}
                                  {diagResults.pytorch?.gpu_name} ·{" "}
                                  {diagResults.pytorch?.gpu_memory_gb} GB
                                </span>
                              ) : (
                                <Tag type="warm-gray" size="sm">CPU only</Tag>
                              )}
                            </div>
                          )}
                        </DiagnosticItem>

                        <DiagnosticItem
                          ok={diagResults.ffmpeg?.ok}
                          label="ffmpeg"
                          value={diagResults.ffmpeg?.ok ? "installed" : "not found"}
                        />

                        <DiagnosticItem
                          ok={diagResults.models?.ok}
                          label="SAM2 Models"
                          value={`${diagResults.models?.count} available`}
                        >
                          {diagResults.models?.available?.length > 0 && (
                            <div style={{ fontSize: "0.75rem", marginTop: "0.25rem", color: "#525252" }}>
                              {diagResults.models.available.join(", ")}
                            </div>
                          )}
                        </DiagnosticItem>

                        <div
                          style={{
                            marginTop: "1rem",
                            padding: "1rem",
                            borderRadius: "4px",
                            textAlign: "center",
                            fontWeight: 600,
                            backgroundColor: diagResults.all_ok ? "#defbe6" : "#fff8e1",
                            color: diagResults.all_ok ? "#198038" : "#8a6d3b",
                          }}
                        >
                          {diagResults.all_ok
                            ? "All systems ready!"
                            : "Some issues detected - check above"}
                        </div>
                      </div>
                    )}

                    {!diagResults && !diagError && !diagLoading && (
                      <p style={{ fontSize: "0.875rem", color: "#6f6f6f" }}>
                        Click "Run Diagnostics" to verify your setup.
                      </p>
                    )}
                  </div>
                </AccordionItem>
              </Accordion>
            </Column>
          </Grid>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid #e0e0e0",
          padding: "1rem",
          textAlign: "center",
          fontSize: "0.875rem",
          color: "#525252",
        }}
      >
        EnvisionBox Project · Donders Institute, Radboud University
      </footer>
    </div>
  );
}
