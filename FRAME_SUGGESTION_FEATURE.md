# Frame Suggestion Feature

## Overview
Automatically suggests optimal frames for annotation based on image quality metrics and AI-powered content analysis.

## Implementation

### Backend

**New Files:**
- `backend/app/frame_analysis.py` - Core frame analysis utilities

**Modified Files:**
- `backend/app/main.py` - Added `/frames/suggest/{session_id}` endpoint
- `backend/app/schemas.py` - Added `SuggestedFrame` and `FrameSuggestionResponse` models

### Frontend

**Modified Files:**
- `frontend/src/api.js` - Added `suggestFrames()` client function
- `frontend/src/pages/AnnotationPage.jsx` - Added frame suggestion UI and multiframe selection

## Features

### 1. Hybrid Scoring System

**Basic Metrics (Always Available):**
- **Sharpness**: Laplacian variance for blur detection
- **Brightness**: Optimal mid-range brightness scoring
- **Edge Density**: Amount of detail/edges in frame
- **Color Variance**: Content diversity via HSV analysis

**DINOv2 Enhancement (Optional):**
- **Objectness Detection**: Identifies frames with distinct objects
- **Visual Diversity**: Ensures suggested frames are different
- **Graceful Fallback**: Uses basic metrics if DINOv2 unavailable

### 2. API Endpoint

```
GET /frames/suggest/{session_id}?top_k=7&use_dinov2=true
```

**Response:**
```json
{
  "session_id": "...",
  "suggested_frames": [
    {
      "frame_index": 42,
      "score": 0.87,
      "sharpness": 125.3,
      "brightness": 0.65,
      "method": "dinov2"
    }
  ],
  "total_analyzed": 50,
  "method_used": "dinov2"
}
```

### 3. User Interface

**Annotation Page:**
- **"Suggest Optimal Frames" button**: Triggers frame analysis
- **Modal with frame grid**: Shows top 7-10 frames with thumbnails
- **Frame metrics display**: Score, sharpness, brightness, method (AI/Basic)
- **Multiframe selection**: Checkbox selection for multiple frames
- **Action buttons**:
  - "Go to Selected": Navigate to first selected frame
  - "Start Annotating (N)": Begin annotating selected frames

## Algorithm

```
1. Sample frames evenly (max 50 for performance)
2. Calculate basic scores for all sampled frames
3. Select top 20 candidates
4. If DINOv2 available:
   a. Calculate objectness scores
   b. Extract CLS embeddings
   c. Ensure frame diversity (cosine distance > 0.3)
   d. Re-rank frames
5. Return top 7 frames
```

## Usage Workflow

1. **Upload video** and **extract frames** (Config page)
2. **Navigate to Annotation page**
3. **Click "Suggest Optimal Frames"**
4. **Wait for analysis** (basic: ~1s, DINOv2: ~5s for 50 frames)
5. **Review suggestions** in modal
6. **Select single or multiple frames**
7. **Click "Start Annotating"**
8. **Annotate selected frames** using point prompts

## Configuration

**Customize suggestions:**
```python
# In frame_analysis.py
suggest_optimal_frames(
    frames_dir=frames_dir,
    top_k=7,              # Number of frames to suggest
    use_dinov2=True,      # Enable AI enhancement
    max_samples=50        # Max frames to analyze
)
```

## Dependencies

**Required:**
- opencv-python
- numpy
- torch
- torchvision

**Optional:**
- DINOv2 (auto-downloaded via torch.hub)

## Performance

**Basic Mode:**
- ~20ms per frame
- Total: ~1 second for 50 frames

**DINOv2 Mode (GPU):**
- ~100ms per frame (initial analysis)
- ~5 seconds for 50 frames
- Requires CUDA or MPS device

**Memory:**
- Basic: <100MB
- DINOv2: ~500MB (model + processing)

## Technical Details

### Frame Diversity Algorithm

Uses cosine similarity on DINOv2 CLS embeddings:
```python
similarity = dot(embedding1_normalized, embedding2_normalized)
distance = 1.0 - similarity

# Ensure minimum distance of 0.3 between selected frames
if distance < 0.3:
    skip_frame()  # Too similar
```

### Scoring Formula

**Basic:**
```python
score = (
    0.4 * normalized_sharpness +
    0.2 * brightness_score +
    0.2 * edge_density +
    0.2 * normalized_color_variance
)
```

**DINOv2 Enhanced:**
```python
final_score = 0.5 * basic_score + 0.5 * normalized_objectness
```

## Troubleshooting

**"DINOv2 enhancement failed, falling back to basic scoring"**
- DINOv2 requires GPU (CUDA/MPS)
- Check internet connection (first-time model download)
- Falls back gracefully to basic metrics

**"No frames could be analyzed"**
- Ensure frames are extracted
- Check frames directory exists
- Verify frame files are valid JPEGs

**Slow analysis (>10s)**
- Reduce `max_samples` (default: 50)
- Disable DINOv2 (`use_dinov2=False`)
- Use CPU-only mode for basic metrics

## Future Enhancements

- [ ] Cache frame suggestions per session
- [ ] Allow user to adjust scoring weights
- [ ] Add temporal smoothness metric (avoid motion blur)
- [ ] Support custom frame sampling strategies
- [ ] Add visual heatmap for objectness
- [ ] Export suggestions to CSV

## Testing

See test instructions in main README.

## Credits

- **SAM2**: Meta AI
- **DINOv2**: Meta AI Research
- **Frame Analysis**: Hybrid implementation combining classical CV and modern AI
