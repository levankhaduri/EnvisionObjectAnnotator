"""Resource profiler for tracking GPU/CPU/RAM usage during processing.

Runs in a background thread and samples system resources at regular intervals.
Saves results as CSV and generates a standalone HTML chart for sharing.
"""

import csv
import json
import threading
import time
from pathlib import Path
from typing import Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class ResourceProfiler:
    """Samples GPU/CPU/RAM at regular intervals during processing."""

    def __init__(
        self,
        output_dir: Path,
        interval_seconds: float = 2.0,
        session_id: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.interval = interval_seconds
        self.session_id = session_id
        self.samples: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start sampling in a background thread."""
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Path:
        """Stop sampling and write results. Returns path to HTML report."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self._save()

    def _sample_loop(self) -> None:
        """Background thread: sample resources at regular intervals."""
        while not self._stop_event.is_set():
            sample = self._take_sample()
            self.samples.append(sample)
            self._stop_event.wait(timeout=self.interval)

    def _take_sample(self) -> dict:
        """Take a single resource snapshot."""
        elapsed = time.perf_counter() - self._start_time
        sample: dict = {"elapsed_s": round(elapsed, 1)}

        # CPU
        if psutil is not None:
            try:
                sample["cpu_percent"] = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                sample["ram_used_gb"] = round(mem.used / 1024**3, 2)
                sample["ram_total_gb"] = round(mem.total / 1024**3, 2)
                sample["ram_percent"] = mem.percent
            except Exception:
                pass

        # GPU
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                sample["gpu_allocated_gb"] = round(allocated, 2)
                sample["gpu_reserved_gb"] = round(reserved, 2)
                sample["gpu_total_gb"] = round(total, 2)
                sample["gpu_percent"] = round((reserved / total) * 100, 1) if total > 0 else 0
                # GPU utilization % (requires pynvml via torch)
                try:
                    utilization = torch.cuda.utilization(0)
                    sample["gpu_utilization"] = utilization
                except Exception:
                    pass
            except Exception:
                pass

        return sample

    def _save(self) -> Path:
        """Save samples as CSV and generate HTML report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = self.output_dir / "resource_profile.csv"
        if self.samples:
            fieldnames = list(self.samples[0].keys())
            # Gather all fields across all samples
            for s in self.samples:
                for k in s:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.samples)

        # Save JSON (for programmatic access)
        json_path = self.output_dir / "resource_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "sample_count": len(self.samples),
                    "interval_seconds": self.interval,
                    "samples": self.samples,
                },
                f,
                indent=2,
            )

        # Generate standalone HTML chart
        html_path = self.output_dir / "resource_profile.html"
        html_path.write_text(self._generate_html(), encoding="utf-8")

        return html_path

    def _generate_html(self) -> str:
        """Generate a standalone HTML page with charts."""
        timestamps = [s.get("elapsed_s", 0) for s in self.samples]
        cpu = [s.get("cpu_percent", 0) for s in self.samples]
        ram_pct = [s.get("ram_percent", 0) for s in self.samples]
        ram_used = [s.get("ram_used_gb", 0) for s in self.samples]
        gpu_pct = [s.get("gpu_percent", 0) for s in self.samples]
        gpu_util = [s.get("gpu_utilization", 0) for s in self.samples]
        gpu_alloc = [s.get("gpu_allocated_gb", 0) for s in self.samples]

        has_gpu = any(s.get("gpu_total_gb", 0) > 0 for s in self.samples)
        has_gpu_util = any(s.get("gpu_utilization") is not None for s in self.samples)
        gpu_total = self.samples[0].get("gpu_total_gb", 0) if self.samples else 0
        ram_total = self.samples[0].get("ram_total_gb", 0) if self.samples else 0

        duration = timestamps[-1] if timestamps else 0
        avg_cpu = sum(cpu) / len(cpu) if cpu else 0
        avg_gpu_util = sum(gpu_util) / len(gpu_util) if gpu_util else 0
        peak_ram = max(ram_used) if ram_used else 0
        peak_gpu = max(gpu_alloc) if gpu_alloc else 0

        # Escape data for JS
        ts_js = json.dumps(timestamps)
        cpu_js = json.dumps(cpu)
        ram_pct_js = json.dumps(ram_pct)
        ram_used_js = json.dumps(ram_used)
        gpu_pct_js = json.dumps(gpu_pct)
        gpu_util_js = json.dumps(gpu_util)
        gpu_alloc_js = json.dumps(gpu_alloc)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Processing Profile — {self.session_id or "Session"}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'IBM Plex Sans', -apple-system, sans-serif; background: #f4f4f4; color: #161616; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
  .meta {{ color: #525252; font-size: 0.875rem; margin-bottom: 1.5rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .stat {{ background: #fff; padding: 1rem; border-left: 3px solid #0f62fe; }}
  .stat-label {{ font-size: 0.75rem; color: #525252; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-value {{ font-size: 1.75rem; font-weight: 600; margin-top: 0.25rem; }}
  .stat-unit {{ font-size: 0.875rem; color: #6f6f6f; }}
  .chart-container {{ background: #fff; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .chart-title {{ font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  canvas {{ width: 100% !important; height: 200px !important; }}
  .footer {{ color: #6f6f6f; font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Processing Resource Profile</h1>
<div class="meta">
  Session: {self.session_id or "N/A"} &mdash;
  Duration: {duration:.0f}s &mdash;
  {len(self.samples)} samples @ {self.interval}s intervals
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-label">Duration</div>
    <div class="stat-value">{duration:.0f}<span class="stat-unit">s</span></div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg CPU</div>
    <div class="stat-value">{avg_cpu:.0f}<span class="stat-unit">%</span></div>
  </div>
  <div class="stat">
    <div class="stat-label">Peak RAM</div>
    <div class="stat-value">{peak_ram:.1f}<span class="stat-unit"> / {ram_total:.0f} GB</span></div>
  </div>
  {"<div class='stat'><div class='stat-label'>Avg GPU Util</div><div class='stat-value'>" + f"{avg_gpu_util:.0f}" + "<span class='stat-unit'>%</span></div></div>" if has_gpu_util else ""}
  {"<div class='stat'><div class='stat-label'>Peak GPU Mem</div><div class='stat-value'>" + f"{peak_gpu:.1f}" + f"<span class='stat-unit'> / {gpu_total:.0f} GB</span></div></div>" if has_gpu else ""}
</div>

<div class="chart-container">
  <div class="chart-title">CPU Usage (%)</div>
  <canvas id="cpuChart"></canvas>
</div>

<div class="chart-container">
  <div class="chart-title">RAM Usage (GB)</div>
  <canvas id="ramChart"></canvas>
</div>

{"<div class='chart-container'><div class='chart-title'>GPU Memory (GB)</div><canvas id='gpuMemChart'></canvas></div>" if has_gpu else ""}

{"<div class='chart-container'><div class='chart-title'>GPU Utilization (%)</div><canvas id='gpuUtilChart'></canvas></div>" if has_gpu_util else ""}

<div class="footer">
  Generated by EnvisionObjectAnnotator Resource Profiler &mdash; {time.strftime('%Y-%m-%d %H:%M:%S')}
</div>

<script>
// Minimal canvas chart renderer (no dependencies)
function drawChart(canvasId, labels, datasets, yMax) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {{ top: 10, right: 20, bottom: 30, left: 50 }};
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  if (!labels.length) return;
  const xMax = Math.max(...labels);
  if (yMax === undefined) {{
    yMax = 0;
    datasets.forEach(d => {{ d.data.forEach(v => {{ if (v > yMax) yMax = v; }}); }});
    yMax = Math.ceil(yMax * 1.1) || 100;
  }}

  // Grid
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + plotH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    ctx.fillStyle = '#6f6f6f'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((yMax * i / 4).toFixed(1), pad.left - 5, y + 4);
  }}

  // X axis labels
  ctx.fillStyle = '#6f6f6f'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
  const xSteps = Math.min(8, labels.length);
  for (let i = 0; i < xSteps; i++) {{
    const idx = Math.floor(i * (labels.length - 1) / Math.max(1, xSteps - 1));
    const x = pad.left + (labels[idx] / (xMax || 1)) * plotW;
    ctx.fillText(labels[idx].toFixed(0) + 's', x, H - 5);
  }}

  // Data lines
  const colors = ['#0f62fe', '#da1e28', '#198038', '#8a3ffc'];
  datasets.forEach((ds, di) => {{
    ctx.strokeStyle = colors[di % colors.length];
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ds.data.forEach((v, i) => {{
      const x = pad.left + (labels[i] / (xMax || 1)) * plotW;
      const y = pad.top + plotH * (1 - v / (yMax || 1));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // Fill
    ctx.globalAlpha = 0.1;
    ctx.lineTo(pad.left + (labels[labels.length - 1] / (xMax || 1)) * plotW, pad.top + plotH);
    ctx.lineTo(pad.left + (labels[0] / (xMax || 1)) * plotW, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = colors[di % colors.length];
    ctx.fill();
    ctx.globalAlpha = 1;

    // Legend
    const lx = pad.left + 10 + di * 120;
    ctx.fillStyle = colors[di % colors.length];
    ctx.fillRect(lx, pad.top, 12, 3);
    ctx.fillStyle = '#161616'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(ds.label, lx + 16, pad.top + 5);
  }});
}}

const timestamps = {ts_js};
drawChart('cpuChart', timestamps, [{{label: 'CPU %', data: {cpu_js}}}], 100);
drawChart('ramChart', timestamps, [{{label: 'Used GB', data: {ram_used_js}}}], {ram_total or 32});
{"drawChart('gpuMemChart', timestamps, [{label: 'Allocated', data: " + gpu_alloc_js + "}, {label: 'Reserved', data: " + gpu_pct_js.replace('[', '[').replace(']', ']') + "}], " + str(gpu_total or 16) + ");" if has_gpu else ""}
{"drawChart('gpuUtilChart', timestamps, [{label: 'GPU Util %', data: " + gpu_util_js + "}], 100);" if has_gpu_util else ""}

// Redraw on resize
window.addEventListener('resize', () => {{
  drawChart('cpuChart', timestamps, [{{label: 'CPU %', data: {cpu_js}}}], 100);
  drawChart('ramChart', timestamps, [{{label: 'Used GB', data: {ram_used_js}}}], {ram_total or 32});
  {"drawChart('gpuMemChart', timestamps, [{label: 'Allocated', data: " + gpu_alloc_js + "}, {label: 'Reserved', data: " + gpu_pct_js + "}], " + str(gpu_total or 16) + ");" if has_gpu else ""}
  {"drawChart('gpuUtilChart', timestamps, [{label: 'GPU Util %', data: " + gpu_util_js + "}], 100);" if has_gpu_util else ""}
}});
</script>
</body>
</html>"""
