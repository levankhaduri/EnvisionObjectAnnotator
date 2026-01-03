import React from "react";
import { Link, useNavigate } from "react-router-dom";

export default function IntroPage() {
  const navigate = useNavigate();

  function scrollToId(id) {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  return (
    <div className="bg-white text-black">
      <style>{`
        body { font-family: 'Inter', sans-serif; }
        .card { background: white; border: 2px solid #e5e7eb; border-radius: 12px; padding: 32px; transition: all 0.3s ease; }
        .card:hover { border-color: #d1d5db; }
        .btn-primary { background: black; color: white; padding: 14px 32px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; }
        .btn-primary:hover { background: #374151; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .btn-secondary { background: white; color: black; border: 2px solid black; padding: 12px 28px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; }
        .btn-secondary:hover { background: black; color: white; }
        .feature-icon { width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 16px; }
        .step-number { width: 40px; height: 40px; border-radius: 50%; background: black; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; }
        .checklist-item { padding: 16px; border: 2px solid #e5e7eb; border-radius: 8px; transition: all 0.2s ease; }
        .checklist-item:hover { border-color: black; }
        .checklist-item input[type="checkbox"]:checked + label { text-decoration: line-through; color: #9ca3af; }
        .requirement-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 8px; }
        .badge-required { background: #fee2e2; color: #991b1b; }
        .badge-recommended { background: #fef3c7; color: #92400e; }
        input[type="checkbox"] { accent-color: black; }
      `}</style>

      <header className="bg-white border-b-2 border-black fixed w-full z-50 top-0">
        <nav className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-black flex items-center justify-center">
                <i className="fas fa-eye text-white text-xl"></i>
              </div>
              <div>
                <div className="text-xl font-bold">EnvisionObjectAnnotator</div>
                <div className="text-xs text-gray-600">Powered by SAM2</div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/DavAhm/EnvisionObjectAnnotator"
                target="_blank"
                rel="noreferrer"
                className="text-black hover:text-gray-600"
              >
                <i className="fab fa-github text-2xl"></i>
              </a>
              <button className="btn-secondary text-sm" onClick={() => navigate("/config")}>
                Skip to Setup <i className="fas fa-arrow-right"></i>
              </button>
            </div>
          </div>
        </nav>
      </header>

      <section className="bg-white pt-32 pb-20 border-b-2 border-black">
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl font-bold mb-6">Automatic Object Tracking & Overlap Detection</h1>
            <p className="text-xl mb-8 text-gray-600">
              Leverage SAM2&apos;s powerful segmentation to automatically track objects in videos and detect
              target-object overlaps—perfect for eye-tracking research and behavioral analysis.
            </p>
            <div className="flex justify-center gap-4">
              <button className="btn-primary" onClick={() => scrollToId("setup")}
              >
                <i className="fas fa-rocket"></i>
                Get Started
              </button>
              <button className="btn-secondary" onClick={() => scrollToId("features")}
              >
                <i className="fas fa-info-circle"></i>
                Learn More
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50" id="features">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">What Can EOA Do?</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="card text-center">
              <div className="feature-icon bg-gray-100 text-black mx-auto">
                <i className="fas fa-search-plus"></i>
              </div>
              <h3 className="text-xl font-bold mb-3">Automatic Object Tracking</h3>
              <p className="text-gray-600">
                Mark objects once on a reference frame, and SAM2 tracks them throughout your entire video automatically.
              </p>
            </div>
            <div className="card text-center">
              <div className="feature-icon bg-gray-100 text-black mx-auto">
                <i className="fas fa-crosshairs"></i>
              </div>
              <h3 className="text-xl font-bold mb-3">Overlap Detection</h3>
              <p className="text-gray-600">
                Detects when targets (e.g., gaze markers) overlap with objects using pixel overlap and centroid-based logic.
              </p>
            </div>
            <div className="card text-center">
              <div className="feature-icon bg-gray-100 text-black mx-auto">
                <i className="fas fa-file-export"></i>
              </div>
              <h3 className="text-xl font-bold mb-3">Multi-Format Export</h3>
              <p className="text-gray-600">Get annotated videos, CSV data files, and ELAN behavioral coding files ready for analysis.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-white border-b-2 border-black">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Perfect For</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            <div className="card">
              <div className="flex items-start gap-4">
                <i className="fas fa-baby text-3xl mt-1"></i>
                <div>
                  <h4 className="font-bold text-lg mb-2">Eye-Tracking Research</h4>
                  <p className="text-gray-600">Analyze infant-caregiver interactions, gaze patterns, and attention to objects in first-person POV videos.</p>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="flex items-start gap-4">
                <i className="fas fa-users text-3xl mt-1"></i>
                <div>
                  <h4 className="font-bold text-lg mb-2">Social Interaction Studies</h4>
                  <p className="text-gray-600">Track objects in social contexts and detect object-directed behaviors automatically.</p>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="flex items-start gap-4">
                <i className="fas fa-brain text-3xl mt-1"></i>
                <div>
                  <h4 className="font-bold text-lg mb-2">Behavioral Coding</h4>
                  <p className="text-gray-600">Replace time-consuming manual coding with automated detection and timeline generation.</p>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="flex items-start gap-4">
                <i className="fas fa-chart-line text-3xl mt-1"></i>
                <div>
                  <h4 className="font-bold text-lg mb-2">Multimodal Analysis</h4>
                  <p className="text-gray-600">Sync with physiology, speech, or movement data using frame-accurate timestamps.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50" id="setup">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">System Requirements</h2>
          <div className="max-w-4xl mx-auto">
            <div className="card mb-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-3">
                <i className="fas fa-server"></i>
                Hardware Requirements
              </h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <span className="requirement-badge badge-required">REQUIRED</span>
                  <div>
                    <strong>Operating System:</strong>
                    <p className="text-gray-600 text-sm">Windows 10 or higher (Linux may work but not covered in this guide)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="requirement-badge badge-recommended">RECOMMENDED</span>
                  <div>
                    <strong>GPU:</strong>
                    <p className="text-gray-600 text-sm">NVIDIA GPU with ≥8 GB VRAM (CPU-only is possible but much slower)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="requirement-badge badge-recommended">RECOMMENDED</span>
                  <div>
                    <strong>RAM:</strong>
                    <p className="text-gray-600 text-sm">16 GB or more for processing large videos</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-3">
                <i className="fas fa-code"></i>
                Software Requirements
              </h3>
              <div className="space-y-4">
                {[
                  {
                    id: "check1",
                    title: "Python 3.10+",
                    desc: "Required for SAM2 and backend API",
                    link: "https://www.python.org/downloads/",
                  },
                  {
                    id: "check2",
                    title: "Node.js 18+",
                    desc: "For the React frontend",
                    link: "https://nodejs.org/en/download",
                  },
                  {
                    id: "check3",
                    title: "SAM2 Repository",
                    desc: "Core segmentation model",
                    link: "https://github.com/DavAhm/EnvisionObjectAnnotator/blob/main/docs/installation_SAM2.md",
                  },
                  {
                    id: "check4",
                    title: "ffmpeg",
                    desc: "For video frame extraction and processing",
                    link: "https://ffmpeg.org/download.html",
                  },
                  {
                    id: "check5",
                    title: "EnvisionObjectAnnotator",
                    desc: "Main application repository",
                    link: "https://github.com/DavAhm/EnvisionObjectAnnotator",
                  },
                ].map((item) => (
                  <div className="checklist-item" key={item.id}>
                    <div className="flex items-start gap-3">
                      <input type="checkbox" id={item.id} className="w-5 h-5 mt-1" />
                      <label htmlFor={item.id} className="flex-1 cursor-pointer">
                        <strong>{item.title}</strong>
                        <p className="text-sm text-gray-600">{item.desc}</p>
                        <a href={item.link} target="_blank" rel="noreferrer" className="text-black underline text-sm hover:text-gray-600">
                          {item.title === "EnvisionObjectAnnotator" ? "Clone Repository →" : "Download / Installation →"}
                        </a>
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-white border-b-2 border-black">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Installation Guide</h2>
          <div className="max-w-4xl mx-auto space-y-6">
            {[
              {
                step: 1,
                title: "Clone the Repository",
                text: "Download the EnvisionObjectAnnotator code from GitHub",
                code: "git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git\ncd EnvisionObjectAnnotator",
              },
              {
                step: 2,
                title: "Install Python + Node",
                text: "Ensure Python 3.10+ and Node.js 18+ are installed",
                code: "python --version\nnode --version",
              },
              {
                step: 3,
                title: "Set Up Backend Environment",
                text: "Create a virtual environment and install backend dependencies",
                code: "cd backend\npython3 -m venv .venv\n./.venv/bin/pip install -r requirements.txt\n./.venv/bin/pip install numpy matplotlib tqdm opencv-python torch torchvision torchaudio",
              },
              {
                step: 4,
                title: "Install SAM2",
                text: "Clone SAM2 and install it into the backend venv",
                code: "cd sam2\n../backend/.venv/bin/pip install -e .",
              },
              {
                step: 5,
                title: "Download Model Checkpoints",
                text: "Download pre-trained SAM2 weights",
                code: "cd checkpoints\n./download_ckpts.sh",
              },
              {
                step: 6,
                title: "Install Frontend Dependencies",
                text: "Install packages for the React UI",
                code: "cd frontend\nnpm install",
              },
            ].map((item) => (
              <div className="card" key={item.step}>
                <div className="flex gap-4">
                  <div className="step-number flex-shrink-0">{item.step}</div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-3">{item.title}</h3>
                    <p className="text-gray-600 mb-3">{item.text}</p>
                    <div className="bg-black text-white p-4 rounded-lg font-mono text-sm overflow-x-auto whitespace-pre-line">
                      {item.code}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            <div className="card border-black">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <i className="fas fa-check-circle"></i>
                Verify Installation
              </h3>
              <p className="text-gray-700 mb-4">Run these commands to ensure everything is installed correctly:</p>
              <div className="space-y-3">
                <div>
                  <p className="text-sm font-semibold mb-1">Check PyTorch:</p>
                  <div className="bg-black text-white p-3 rounded-lg font-mono text-sm">
                    python -c "import torch; print(torch.__version__)"
                  </div>
                  <p className="text-xs text-gray-600 mt-1">Expected: PyTorch version (e.g., 2.1.0)</p>
                </div>
                <div>
                  <p className="text-sm font-semibold mb-1">Check SAM2:</p>
                  <div className="bg-black text-white p-3 rounded-lg font-mono text-sm">
                    python -c "import sam2; print('SAM2 installed successfully')"
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold mb-1">Check ffmpeg:</p>
                  <div className="bg-black text-white p-3 rounded-lg font-mono text-sm">
                    ffmpeg -version
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Launch Application</h2>
          <div className="max-w-3xl mx-auto">
            <div className="card">
              <div className="text-center mb-6">
                <i className="fas fa-rocket text-6xl mb-4"></i>
                <h3 className="text-2xl font-bold mb-2">Ready to Start!</h3>
                <p className="text-gray-600">Once installation is complete, launch EOA with these commands:</p>
              </div>
              <div className="bg-black text-white p-6 rounded-lg font-mono text-sm mb-6 whitespace-pre-line">
                cd backend
                {"\n"}./.venv/bin/uvicorn app.main:app --reload
                {"\n\n"}cd frontend
                {"\n"}npm run dev
              </div>
              <div className="bg-gray-100 border-2 border-gray-300 p-4 rounded-lg mb-6">
                <p className="text-sm"><strong>Note:</strong> The API runs on port 8000 and the UI runs on port 5173 by default.</p>
              </div>
              <div className="text-center">
                <button className="btn-primary" onClick={() => navigate("/config")}
                >
                  Continue to Configuration
                  <i className="fas fa-arrow-right"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="bg-black text-white py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <div>
              <h4 className="font-bold mb-4">EnvisionObjectAnnotator</h4>
              <p className="text-gray-400 text-sm">Automatic object tracking and overlap detection powered by SAM2.</p>
            </div>
            <div>
              <h4 className="font-bold mb-4">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="https://github.com/DavAhm/EnvisionObjectAnnotator" className="hover:text-white">GitHub Repository</a></li>
                <li><a href="https://github.com/DavAhm/EnvisionObjectAnnotator/blob/main/docs/installation_SAM2.md" className="hover:text-white">Installation Guide</a></li>
                <li><a href="#" className="hover:text-white">Documentation</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-4">Contact</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Davide Ahmar</li>
                <li>Babajide Owoyele</li>
                <li>Wim Pouw</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 pt-6 text-center text-sm text-gray-400">
            <p>Funded by Donders Research Stimulation Fund, Radboud University, 2025</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
