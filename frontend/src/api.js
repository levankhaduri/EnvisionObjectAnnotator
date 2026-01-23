export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function createSession(name) {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    throw new Error("Failed to create session");
  }
  return res.json();
}

export async function uploadVideo(sessionId, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/uploads/video?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Failed to upload video");
  }
  return res.json();
}

export async function updateConfig(payload) {
  const res = await fetch(`${API_BASE}/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to update config");
  }
  return res.json();
}

export async function extractFrames(sessionId, quality = 2) {
  const res = await fetch(`${API_BASE}/frames/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, quality }),
  });
  if (!res.ok) {
    throw new Error("Failed to extract frames");
  }
  return res.json();
}

export async function listFrames(sessionId) {
  const res = await fetch(`${API_BASE}/frames/list/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to load frames");
  }
  return res.json();
}

export async function submitAnnotation(payload) {
  const res = await fetch(`${API_BASE}/annotation/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to save annotation points");
  }
  return res.json();
}

export async function testMask(payload) {
  const res = await fetch(`${API_BASE}/annotation/test-mask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to test mask");
  }
  return res.json();
}

export async function fetchFrameAnnotations(sessionId, frameIndex) {
  const res = await fetch(`${API_BASE}/annotation/frames/${sessionId}/${frameIndex}`);
  if (!res.ok) {
    throw new Error("Failed to load annotations");
  }
  return res.json();
}

export async function startProcessing(sessionId) {
  const res = await fetch(`${API_BASE}/processing/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    throw new Error("Failed to start processing");
  }
  return res.json();
}

export async function fetchProcessingStatus(sessionId) {
  const res = await fetch(`${API_BASE}/processing/status/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch processing status");
  }
  return res.json();
}

export function getProcessingPreviewUrl(sessionId) {
  return `${API_BASE}/processing/preview/${sessionId}`;
}

export async function fetchResults(sessionId) {
  const res = await fetch(`${API_BASE}/results/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch results");
  }
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    throw new Error("Failed to fetch models");
  }
  return res.json();
}

export function getResultDownloadUrl(sessionId, kind) {
  const params = new URLSearchParams({ kind });
  return `${API_BASE}/results/download/${sessionId}?${params.toString()}`;
}
