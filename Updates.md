# Updates — TASK-CREATEFRONTEND

1. Added delete functionality for annotated objects (backend endpoint + frontend X button)
2. Moved "Suggest Optimal Frames" button from left column to right column, above Shortcuts
3. Wrapped right panel in a scrollable sticky container so it follows the viewport
4. Bidirectional propagation now enabled by default in frontend and backend
5. Live progress bar during SAM2 propagation (shows frame count instead of stuck at 20%)
6. Processing error status now shown on ProcessingPage with link to partial results
7. ResultsPage download buttons fixed (disabled state works, downloads open in new tab)
8. Backend results endpoint now checks actual file existence on disk
