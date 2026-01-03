import React from "react";
import { Link } from "react-router-dom";

export default function PlaceholderPage({ title }) {
  return (
    <div className="page-shell">
      <header className="simple-header">
        <Link className="brand" to="/">
          <span className="brand-mark">EOA</span>
          <span className="brand-text">EnvisionObjectAnnotator</span>
        </Link>
        <nav className="simple-nav">
          <Link to="/config">Config</Link>
          <Link to="/annotation">Annotation</Link>
          <Link to="/processing">Processing</Link>
          <Link to="/results">Results</Link>
        </nav>
      </header>
      <main className="placeholder">
        <div className="placeholder-card">
          <h1>{title}</h1>
          <p>UI coming next. This page will connect to the backend API.</p>
          <Link className="btn btn-primary" to="/">
            Back to Intro
          </Link>
        </div>
      </main>
    </div>
  );
}
