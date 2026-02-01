# Bidirectional Propagation Feature

## Overview
Enables SAM2 to track objects both forward AND backward from annotated frames, providing complete video coverage and improved tracking accuracy for multiframe selections.

## Implementation

### What Changed

**Backend Changes:**
- [pipeline.py](backend/app/pipeline.py): Added `enable_bidirectional` parameter to `UltraOptimizedProcessor`
- [processing.py](backend/app/processing.py):
  - Added `_load_multiframe_annotations()` for loading annotations from all frames
  - Modified `run_processing()` to detect and handle multiframe scenarios
  - Enabled `enable_bidirectional=True` by default

### How It Works

#### Single Reference Frame
When you annotate objects on a single frame (e.g., frame 50):

**Before** (Forward only):
- Tracks objects from frame 50 → end
- Frames 0-49 have NO tracking

**After** (Bidirectional):
- **Forward pass**: Tracks from frame 50 → end
- **Backward pass**: Tracks from frame 50 → 0
- **Result**: Complete coverage of entire video

#### Multiple Reference Frames
When you use the "Suggest Optimal Frames" feature and annotate multiple frames (e.g., frames 10, 50, 100):

1. **Global Object Mapping**: Same object name across frames gets same ID
   - "dog" on frame 10 → ID 1
   - "dog" on frame 50 → ID 1 (same object)
   - "cat" on frame 10 → ID 2

2. **Bidirectional from Each Frame**: Each annotated frame serves as a reference point for bidirectional propagation

3. **Better Coverage**: Gaps between annotated frames are filled more accurately

## Key Benefits

### 1. Complete Video Coverage
- No more "dead zones" before your first annotation
- Backward propagation fills in missing frames

### 2. Improved Multiframe Tracking
- When objects are annotated on multiple frames, tracking is more robust
- Reduces drift and improves accuracy across frame gaps

### 3. Consistent Object IDs
- Same object name = same ID across all frames
- Makes it easier to track the same object throughout the video

## Technical Details

### SAM2 Reverse Propagation
Uses SAM2's built-in `reverse=True` parameter:
```python
# Forward propagation
predictor.propagate_in_video(
    inference_state,
    start_frame_idx=ref_frame,
    reverse=False  # default
)

# Backward propagation
predictor.propagate_in_video(
    inference_state,
    start_frame_idx=ref_frame,
    reverse=True  # process frames in reverse order
)
```

### Processing Flow

1. **Load Annotations**: Detect all annotated frames
2. **Initialize SAM2**: Build video predictor with selected model
3. **Add Prompts**: Register point prompts for annotated objects
4. **Forward Pass**: Propagate from reference frame to video end
5. **Reset State**: Clear inference state for backward pass
6. **Re-add Prompts**: Re-register the same point prompts
7. **Backward Pass**: Propagate from reference frame to video start
8. **Merge Results**: Combine forward and backward masks
9. **Export**: Generate video, ELAN, and CSV outputs

### Performance Impact

**Memory**: ~1.5x usage (two propagation passes)
- Forward pass stores masks
- Backward pass adds more masks
- Disk-backed storage kicks in for long videos

**Time**: ~2x processing time
- Effectively processing the video twice
- Still faster than manual annotation

**Quality**: Significantly better tracking
- Fewer gaps and lost tracks
- More robust to occlusions
- Better handling of objects entering/exiting frame

## Usage

### Basic Workflow

1. **Upload Video** → Extract frames (Config page)
2. **Suggest Optimal Frames** (Annotation page)
3. **Select Multiple Frames** from suggestions
4. **Annotate Objects** on selected frames using point prompts
5. **Start Processing** → Bidirectional propagation runs automatically
6. **View Results** → Check output video with complete tracking

### Example Scenario

**Problem**: Tracking a bird that flies into frame mid-video
- Without bidirectional: Only tracks from first annotation forward
- With bidirectional: Tracks backward too, catching earlier appearances

**Solution**:
1. Annotate the bird on frame 100 (where it's clearly visible)
2. Bidirectional propagation:
   - Forward: Frame 100 → end
   - Backward: Frame 100 → 0
3. Result: Bird is tracked even if it appeared before frame 100

## Configuration

Bidirectional propagation is **enabled by default**. To disable:

```python
# In processing.py, line ~708
processor = UltraOptimizedProcessor(
    ...
    enable_bidirectional=False,  # Disable backward pass
)
```

## Troubleshooting

**"Out of Memory" errors:**
- Bidirectional uses more memory (two passes)
- Automatic fallbacks: Reduced batch size, disk-backed storage
- For very long videos: Consider using frame stride or ROI features

**Tracking quality issues:**
- Ensure annotated frames have clear, distinct objects
- Use "Suggest Optimal Frames" to find best frames
- Add more reference frames in challenging sections

**Slow processing:**
- Normal - bidirectional takes ~2x time
- Monitor progress in Processing Status page
- Use smaller model (hiera_tiny) for faster processing

## Future Enhancements

- [ ] Track merging across multiple reference frames
- [ ] Adaptive bidirectional (only when needed)
- [ ] Per-object bidirectional control
- [ ] Visualization of forward vs backward tracks
- [ ] Smart conflict resolution for overlapping tracks

## Testing

To test bidirectional propagation:

1. Create a simple 30-second video
2. Annotate an object on frame 100 (mid-video)
3. Process with bidirectional enabled
4. Check output video - object should be tracked from frame 0 to end
5. Compare with bidirectional disabled - object only tracked frame 100 onward

## Technical Notes

### Object ID Assignment
Objects are assigned IDs based on sorted object names:
```python
object_name_to_id = {
    "cat": 1,
    "dog": 2,
    "target_person": 3,
}
```

This ensures consistent IDs across frames when the same objects are annotated multiple times.

### Memory Management
- Uses same disk-backed storage as forward propagation
- Automatically offloads masks to disk under memory pressure
- Each pass (forward/backward) shares the same storage infrastructure

### Overlap Detection
- Target overlap detection works seamlessly with bidirectional
- "Looking at" events detected in both forward and backward frames
- ELAN output includes all detected events across entire video

## Credits
- **SAM2**: Meta AI (supports `reverse=True` parameter)
- **Implementation**: Bidirectional propagation with multiframe support
- **Feature Request**: Improved multiframe selection tracking
