import React from "react";
import { Routes, Route } from "react-router-dom";
import IntroPage from "./pages/IntroPage.jsx";
import ConfigPage from "./pages/ConfigPage.jsx";
import AnnotationPage from "./pages/AnnotationPage.jsx";
import ProcessingPage from "./pages/ProcessingPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import PlaceholderPage from "./pages/PlaceholderPage.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<IntroPage />} />
      <Route path="/config" element={<ConfigPage />} />
      <Route path="/annotation" element={<AnnotationPage />} />
      <Route path="/processing" element={<ProcessingPage />} />
      <Route path="/results" element={<ResultsPage />} />
    </Routes>
  );
}
