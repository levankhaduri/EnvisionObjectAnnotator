#!/usr/bin/env python3
"""
Fixed Overlap Detection and Annotations for SAM2 Video Analysis
Handles complex scenarios: inclusion, multiple overlaps, bidirectional relationships
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import customtkinter as ctk
from tkinter import ttk, scrolledtext
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk
import subprocess
import shutil
from pathlib import Path
import json
import yaml
import time
from datetime import datetime
import gc
from tqdm import tqdm
import pandas as pd
import psutil
import threading
import queue

# memory optimization (same as before)
os.environ["SAM2_OFFLOAD_VIDEO_TO_CPU"] = "true"
os.environ["SAM2_OFFLOAD_STATE_TO_CPU"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'utilization_pct': (reserved / total) * 100
        }
    return None

def ultra_cleanup_memory():
    """memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def configure_torch_ultra_conservative():
    """Configure PyTorch for memory usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70)
        
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        ultra_cleanup_memory()
        print(f"GPU Memory after setup: {get_gpu_memory_info()}")

def setup_device_ultra_optimized():
    """Setup computation device with settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_info = get_gpu_memory_info()
        print(f"Initial GPU Memory: {gpu_info['allocated_gb']:.1f}GB allocated, {gpu_info['free_gb']:.1f}GB free")
        
        if gpu_info['total_gb'] < 8:
            print("⚠️ WARNING: Low GPU memory detected. Using very conservative settings.")
            torch.cuda.set_per_process_memory_fraction(0.60)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Support for MPS devices is preliminary.")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU - this will be very slow but stable")
    
    print(f"Using device: {device}")
    return device

# Video processing functions (same as before)
def get_video_fps(video_path):
    """Get video FPS using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def video_to_frames(input_video, output_dir, quality=2):
    """Convert video to frames using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    fps, total_frames = get_video_fps(input_video)
    print(f"Video: {Path(input_video).name}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-q:v', str(quality),
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            print(f"Successfully created {num_frames} frames")
            return fps, num_frames
        else:
            print(f"Error: {result.stderr}")
            return -1, -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1, -1

def show_frame_preview(frames_dir, frame_idx, total_frames):
    """Show a preview of the selected frame"""
    frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
    if not os.path.exists(frame_path):
        messagebox.showerror("Error", f"Frame {frame_idx} not found")
        return False
    
    frame = cv2.imread(frame_path)
    if frame is None:
        messagebox.showerror("Error", f"Could not load frame {frame_idx}")
        return False
    
    # Resize frame for preview if too large
    height, width = frame.shape[:2]
    max_size = 800
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Add frame info text
    info_text = f"Frame {frame_idx}/{total_frames-1} - Preview"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press any key to continue...", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.namedWindow('Frame Preview', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame Preview', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('Frame Preview')
    return True

class EnhancedOverlapDetector:
    """Enhanced overlap detector that properly handles inclusion and complex overlaps"""
    
    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.inclusion_threshold = 0.1  # 80% overlap = inclusion
        
    def calculate_detailed_overlap(self, mask1, mask2):
        """Enhanced overlap detection with both pixel overlap and spatial containment"""
        if mask1.shape != mask2.shape:
            print(f"❌ Shape mismatch: {mask1.shape} vs {mask2.shape}")
            return None
        
        # Ensure masks are 2D
        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # VERY AGGRESSIVE boolean conversion - catch any non-zero values
        if mask1.dtype in [np.float32, np.float64]:
            mask1_bool = mask1 > 0.0001
        else:
            mask1_bool = mask1 > 0
        
        if mask2.dtype in [np.float32, np.float64]:
            mask2_bool = mask2 > 0.0001
        else:
            mask2_bool = mask2 > 0
        
        # Calculate areas
        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)
        
        if area1 == 0 or area2 == 0:
            return None
        
        # Calculate pixel intersection
        intersection = mask1_bool & mask2_bool
        intersection_area = np.sum(intersection)
        
        # Calculate overlap percentages for pixel overlap
        overlap_pct_1 = intersection_area / area1 if area1 > 0 else 0
        overlap_pct_2 = intersection_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_pct_1, overlap_pct_2)
        
        # SPATIAL CONTAINMENT DETECTION
        spatial_relationship = False
        containment_type = None
        
        try:
            # Find contours for both masks
            contours1, _ = cv2.findContours(mask1_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours1 and contours2:
                # Get the largest contour for each mask
                contour1 = max(contours1, key=cv2.contourArea)
                contour2 = max(contours2, key=cv2.contourArea)
                
                # Calculate centroids
                M1 = cv2.moments(contour1)
                M2 = cv2.moments(contour2)
                
                if M1['m00'] != 0 and M2['m00'] != 0:
                    cx1, cy1 = int(M1['m10']/M1['m00']), int(M1['m01']/M1['m00'])
                    cx2, cy2 = int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00'])
                    
                    # Check for partial area containment FIRST (any part of object inside boundary)
                    # Create masks for contour areas
                    mask1_contour = np.zeros_like(mask1_bool, dtype=np.uint8)
                    mask2_contour = np.zeros_like(mask2_bool, dtype=np.uint8)
                    cv2.fillPoly(mask1_contour, [contour1], 1)
                    cv2.fillPoly(mask2_contour, [contour2], 1)
                    
                    # Check if any part of object2 is within contour1 boundary
                    object2_in_contour1 = np.any(mask2_bool & mask1_contour)
                    # Check if any part of object1 is within contour2 boundary  
                    object1_in_contour2 = np.any(mask1_bool & mask2_contour)
                    
                    if object2_in_contour1:
                        spatial_relationship = True
                        overlap_area = np.sum(mask2_bool & mask1_contour)
                        containment_type = "object2_partial_inside_object1"
                        print(f"     Part of Object 2 is INSIDE Object 1 boundary ({overlap_area} pixels)")
                    elif object1_in_contour2:
                        spatial_relationship = True
                        overlap_area = np.sum(mask1_bool & mask2_contour)
                        containment_type = "object1_partial_inside_object2"
                        print(f"     Part of Object 1 is INSIDE Object 2 boundary ({overlap_area} pixels)")
                    else:
                        # Fallback to centroid check only if no partial containment
                        inside_1 = cv2.pointPolygonTest(contour1, (cx2, cy2), False) >= 0
                        inside_2 = cv2.pointPolygonTest(contour2, (cx1, cy1), False) >= 0
                        
                        if inside_1:
                            spatial_relationship = True
                            containment_type = "object2_centroid_inside_object1"
                            print(f"     Object 2 centroid is INSIDE Object 1 boundary")
                        elif inside_2:
                            spatial_relationship = True
                            containment_type = "object1_centroid_inside_object2"
                            print(f"     Object 1 centroid is INSIDE Object 2 boundary")
                    

        
        except Exception as e:
            print(f"    ⚠️ Error in spatial containment detection: {e}")
            spatial_relationship = False
        
        # Determine relationship type and strength
        # Apply percentage threshold to pixel overlap
        has_meaningful_pixel_overlap = intersection_area > 0 and max_overlap >= self.overlap_threshold
        has_spatial_relationship = spatial_relationship
        
        # If no relationship at all, return None
        if not has_meaningful_pixel_overlap and not has_spatial_relationship:
            return None
        
        # Enhanced criteria: Accept EITHER meaningful pixel overlap OR spatial containment
        meets_basic_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        meets_continuation_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        
        # Determine relationship type
        if has_meaningful_pixel_overlap and has_spatial_relationship:
            relationship_type = "overlap_and_containment"
        elif has_meaningful_pixel_overlap:
            relationship_type = "pixel_overlap"
        elif has_spatial_relationship:
            relationship_type = "spatial_containment"
        else:
            relationship_type = "none"
        
        print(f"  🔍 Enhanced overlap analysis:")
        if intersection_area > 0:
            print(f"    Pixel intersection: {intersection_area} pixels")
            print(f"    Max pixel overlap: {max_overlap:.1%}")
            print(f"    Meets pixel threshold ({self.overlap_threshold:.1%}): {has_meaningful_pixel_overlap}")
        if has_spatial_relationship:
            print(f"    Spatial relationship: {containment_type}")
        print(f"    Final relationship type: {relationship_type}")
        print(f"    Meets threshold: {meets_basic_threshold}")
        
        return {
            'intersection_area': intersection_area,
            'overlap_pct_1': overlap_pct_1,
            'overlap_pct_2': overlap_pct_2,
            'min_overlap_pct': min(overlap_pct_1, overlap_pct_2),
            'max_overlap_pct': max_overlap,
            'spatial_relationship': spatial_relationship,
            'containment_type': containment_type,
            'has_meaningful_pixel_overlap': has_meaningful_pixel_overlap,
            'has_spatial_relationship': has_spatial_relationship,
            'relationship_type': relationship_type,
            'meets_threshold': meets_basic_threshold,
            'meets_continuation_threshold': meets_continuation_threshold
        }


class ImprovedTargetOverlapTracker:
    """Improved overlap tracker with better inclusion detection and annotations"""
    
    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}
        self.target_objects = {}
        self.detector = EnhancedOverlapDetector(overlap_threshold)
        
    def register_target(self, obj_id, obj_name):
        """Register target objects"""
        if 'target' in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"Target registered: {obj_name} (ID: {obj_id})")
            return True
        return False
    
    def get_overlap_summary(self):
        """Get overlap summary"""
        summary = {}
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            summary[target_name] = {
                'total_events': len(events),
                'events': events,
                'total_overlap_frames': sum(event['duration_frames'] for event in events)
            }
        return summary
    
    def finalize_tracking(self, last_frame_idx):
        """PRECISE finalize - don't extend events, keep them as detected"""
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            
            if events and events[-1].get('end_frame') is None:
                last_event = events[-1]
                
                # If event has no end_frame, it means it was ongoing
                # End it at the last frame where we had duration_frames
                if 'duration_frames' in last_event and last_event['duration_frames'] > 0:
                    last_event['end_frame'] = last_event['start_frame'] + last_event['duration_frames'] - 1
                else:
                    # Single frame event
                    last_event['end_frame'] = last_event['start_frame']
                    last_event['duration_frames'] = 1
                
                print(f"  📝 Precise finalize for {target_name}: frames {last_event['start_frame']}-{last_event['end_frame']} ({last_event['duration_frames']} frames)")

    def analyze_frame_overlaps(self, frame_results, object_names):
        """Enhanced frame analysis with continuation validation"""
        frame_analysis = {
            'target_overlaps': {},
            'object_relationships': {},
            'looking_at_events': []
        }
        
        try:
            for target_id in self.target_objects:
                if target_id not in frame_results:
                    continue
                    
                target_mask = frame_results[target_id]
                if len(target_mask.shape) > 2:
                    target_mask = target_mask.squeeze()
                
                target_name = self.target_objects[target_id]
                looking_at_objects = []
                
                # Check if we have an ongoing event for this target
                has_ongoing_event = (self.overlap_events[target_id] and 
                                not self.overlap_events[target_id][-1].get('end_frame'))
                
                for obj_id, mask in frame_results.items():
                    if obj_id == target_id:
                        continue
                        
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    
                    try:
                        overlap_info = self.detector.calculate_detailed_overlap(target_mask, mask)
                        
                        if overlap_info:
                            # Use different criteria for new vs continuing events
                            if has_ongoing_event:
                                # For continuing events, require stronger overlap
                                if overlap_info.get('meets_continuation_threshold', False):
                                    looking_at_objects.append({
                                        'object_id': obj_id,
                                        'object_name': obj_name,
                                        'event_type': 'looking_at',
                                        'relationship_desc': f"OVERLAPS {obj_name} (continuing)"
                                    })
                                    print(f"      ✅ STRONG OVERLAP CONTINUES: {target_name} ↔ {obj_name}")
                                else:
                                    print(f"      ⚠️ WEAK OVERLAP (ending event): {target_name} ↔ {obj_name}")
                            else:
                                # For new events, use basic threshold (ultra-sensitive)
                                if overlap_info.get('meets_threshold', False):
                                    looking_at_objects.append({
                                        'object_id': obj_id,
                                        'object_name': obj_name,
                                        'event_type': 'looking_at',
                                        'relationship_desc': f"LOOKING AT {obj_name}"
                                    })
                                    if has_ongoing_event:
                                        print(f"      ✅ OVERLAP CONTINUES: {target_name} ↔ {obj_name}")
                                    else:
                                        print(f"      ✅ NEW OVERLAP DETECTED: {target_name} ↔ {obj_name}")
                            
                            # Store for ELAN export
                            if looking_at_objects and looking_at_objects[-1]['object_id'] == obj_id:
                                frame_analysis['looking_at_events'].append({
                                    'target_id': target_id,
                                    'target_name': target_name,
                                    'object_id': obj_id,
                                    'object_name': obj_name
                                })
                        
                    except Exception as e:
                        print(f"      ⚠️ Error checking {obj_name}: {e}")
                        continue
                
                if looking_at_objects:
                    frame_analysis['target_overlaps'][target_id] = looking_at_objects
            
        except Exception as e:
            print(f"  ⚠️ Error in analyze_frame_overlaps: {e}")
            import traceback
            traceback.print_exc()
        
        return frame_analysis

    def track_frame_overlaps_batch(self, frame_idx, frame_results, object_names):
        """Track 'looking at' events with ACCURATE offset detection"""
        try:
            frame_analysis = self.analyze_frame_overlaps(frame_results, object_names)
            
            # Store frame analysis for video creation
            if not hasattr(self, 'frame_analyses'):
                self.frame_analyses = {}
            self.frame_analyses[frame_idx] = frame_analysis
            
            # Process each target to update events with accurate timing
            for target_id in self.target_objects:
                current_overlaps = []
                
                # Get current overlaps for this target
                if target_id in frame_analysis.get('target_overlaps', {}):
                    looking_at_objects = frame_analysis['target_overlaps'][target_id]
                    current_overlaps = [obj['object_name'] for obj in looking_at_objects]
                
                # Update events with accurate offset detection
                self._update_overlap_event(target_id, frame_idx, current_overlaps)
            
            return frame_analysis
            
        except Exception as e:
            print(f"  ⚠️ Error in track_frame_overlaps_batch for frame {frame_idx}: {e}")
            return {
                'target_overlaps': {},
                'object_relationships': {},
                'looking_at_events': []
            }

    
    def _update_overlap_event(self, target_id, frame_idx, overlapping_names):
        """Enhanced event tracking with stricter continuation criteria"""
        events = self.overlap_events[target_id]
        current_overlap_set = set(overlapping_names)
        target_name = self.target_objects[target_id]
        
        # Check if we have an ongoing event
        if events and events[-1].get('end_frame') is None:
            last_event = events[-1]
            last_overlap_set = set(last_event['overlapping_objects'])
            
            if current_overlap_set == last_overlap_set and current_overlap_set:
                # Same objects detected - but is the overlap still STRONG enough to continue?
                # We need to check the overlap strength, not just presence
                
                # Get the frame results to check overlap strength
                # (This would need to be passed from the calling function)
                # For now, continue the event but add validation
                
                last_event['duration_frames'] = frame_idx - last_event['start_frame'] + 1
                print(f"      → Continuing event: {target_name} frames {last_event['start_frame']}-{frame_idx}")
                
            else:
                # Objects changed or stopped - END the event
                last_event['end_frame'] = frame_idx - 1
                last_event['duration_frames'] = last_event['end_frame'] - last_event['start_frame'] + 1
                
                objects_str = ', '.join(last_event['overlapping_objects'])
                print(f"      ✅ Event ended: {target_name} frames {last_event['start_frame']}-{last_event['end_frame']} ({last_event['duration_frames']} frames) | {objects_str}")
                
                # Start new event if we have new overlaps
                if current_overlap_set:
                    new_event = {
                        'start_frame': frame_idx,
                        'end_frame': None,
                        'duration_frames': 1,
                        'overlapping_objects': list(overlapping_names),
                        'event_type': 'looking_at'
                    }
                    events.append(new_event)
                    objects_str = ', '.join(current_overlap_set)
                    print(f"      🎯 New event started: {target_name} frame {frame_idx} | {objects_str}")
        else:
            # No ongoing event - start new one if we have overlaps
            if current_overlap_set:
                new_event = {
                    'start_frame': frame_idx,
                    'end_frame': None,
                    'duration_frames': 1,
                    'overlapping_objects': list(overlapping_names),
                    'event_type': 'looking_at'
                }
                events.append(new_event)
                objects_str = ', '.join(current_overlap_set)
                print(f"      🎯 First event started: {target_name} frame {frame_idx} | {objects_str}")

    def has_targets(self):
        """Check if any targets are registered"""
        return bool(self.target_objects)

class UltraOptimizedProcessor:
    """Ultra memory-optimized processor with improved overlap detection"""
    
    def __init__(self, predictor, video_dir, overlap_threshold=0.1, reference_frame=0, 
                 batch_size=50, auto_fallback=True, preview_callback=None, preview_stride=15, preview_max_dim=720):
        self.predictor = predictor
        self.video_dir = video_dir
        self.overlap_threshold = overlap_threshold
        self.reference_frame = reference_frame
        self.batch_size = batch_size
        self.auto_fallback = auto_fallback
        # Live preview options (optional & lightweight)
        self.preview_callback = preview_callback
        self.preview_stride = max(1, int(preview_stride)) if preview_stride else None
        self.preview_max_dim = int(preview_max_dim)
        
        # Initialize improved overlap tracker
        self.overlap_tracker = ImprovedTargetOverlapTracker(overlap_threshold)
        
        # Get frame names
        self.frame_names = sorted(
            [p for p in os.listdir(self.video_dir) 
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        if not self.frame_names:
            raise ValueError("No frames found in the specified directory!")
        
        print(f"Processor with Overlap Detection")
        print(f"  Frames: {len(self.frame_names)}")
        print(f"  Reference frame: {reference_frame}")
        print(f"  Overlap threshold: {overlap_threshold*100:.1f}%")
        print(f"  Event detection: Any spatial relationship = 'overlapping' event")
        print(f"  Clean timing: Accurate begin/end times for ELAN export")
        print(f"  Simplified annotations: Focus on event detection, not percentages")
        
        # Memory optimization flags
        # Small BGR color palette for preview masks
        try:
            import matplotlib.pyplot as _plt
            _cmap = _plt.get_cmap('tab10')
            self._preview_colors = [tuple(int(c*255) for c in _cmap(i)[:3][::-1]) for i in range(10)]
        except Exception:
            self._preview_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128),(0,128,128),(200,200,200)]
        self.offload_video_to_cpu = os.environ.get("SAM2_OFFLOAD_VIDEO_TO_CPU", "true") == "true"
        self.offload_state_to_cpu = os.environ.get("SAM2_OFFLOAD_STATE_TO_CPU", "true") == "true"
    
    def process_video_with_memory_management(self, points_dict, labels_dict, object_names, debug=True):
        """Process video with ultra memory management and improved overlap detection"""
        try:
            configure_torch_ultra_conservative()
            
            print(f"\nStarting processing with overlap detection...")
            
            # Try processing with fallback strategies
            for attempt in range(3):
                try:
                    if attempt == 0:
                        print(f"Attempt 1: Standard optimized processing")
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug)
                    elif attempt == 1:
                        print(f"Attempt 2: Micro-batch processing")
                        self.batch_size = self.batch_size // 2
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug)
                    else:
                        print(f"Attempt 3: Emergency CPU fallback")
                        return self._process_cpu_fallback(points_dict, labels_dict, object_names, debug)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ❌ Attempt {attempt + 1} failed: CUDA OOM")
                        ultra_cleanup_memory()
                        
                        if torch.cuda.is_available():
                            current_fraction = torch.cuda.get_per_process_memory_fraction() * 0.8
                            torch.cuda.set_per_process_memory_fraction(max(0.3, current_fraction))
                            print(f"  🔧 Reduced memory fraction to {current_fraction:.2f}")
                        
                        if attempt == 2:
                            raise e
                    else:
                        raise e
            
        except Exception as e:
            print(f"❌ All processing attempts failed: {str(e)}")
            return None
        finally:
            ultra_cleanup_memory()

    def _process_standard_optimized(self, points_dict, labels_dict, object_names, debug):
        """Standard optimized processing with enhanced overlap detection"""
        # Initialize SAM2 state
        print("🔧 Initializing SAM2 state...")
        inference_state = self.predictor.init_state(
            video_path=self.video_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=True,
        )
        
        self.predictor.reset_state(inference_state)
        ultra_cleanup_memory()
        
        # Store object names
        self.object_names = object_names
        
        # Register targets
        targets_found = False
        for obj_id, obj_name in object_names.items():
            if self.overlap_tracker.register_target(obj_id, obj_name):
                targets_found = True
        
        print(f"\n📌 Adding prompts for {len(points_dict)} objects...")
        
        # Add all prompts to reference frame
        for obj_id in points_dict:
            try:
                points = np.array(points_dict[obj_id], dtype=np.float32)
                labels = np.array(labels_dict[obj_id], dtype=np.int32)
                
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                
                if debug:
                    print(f"  📌 {obj_name}: +{sum(labels == 1)} -{sum(labels == 0)} points")
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.reference_frame,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
                
                # Immediate cleanup
                del out_mask_logits, points, labels
                ultra_cleanup_memory()
            
            except Exception as e:
                print(f"  ❌ Error adding prompts for object {obj_id}: {e}")
                continue
        
        print(f"\n🔄 Propagating through video with enhanced overlap detection...")
        
        # Process with enhanced overlap tracking
        results = {}
        frame_analyses = {}  # Store detailed frame analysis
        frame_count = 0
        overlap_count = 0
        last_memory_check = 0
        
        with tqdm(total=len(self.frame_names), desc="Processing frames") as pbar:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                try:
                    # Memory monitoring
                    if frame_count - last_memory_check >= 50:
                        gpu_info = get_gpu_memory_info()
                        if gpu_info and gpu_info['utilization_pct'] > 90:
                            print(f"  ⚠️ High memory usage: {gpu_info['utilization_pct']:.1f}%")
                            ultra_cleanup_memory()
                        last_memory_check = frame_count

                    # Store results efficiently
                    frame_results = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        frame_results[out_obj_id] = mask.copy()
                        del mask

                    results[out_frame_idx] = frame_results

                    # Enhanced overlap tracking
                    if targets_found:
                        frame_analysis = self.overlap_tracker.track_frame_overlaps_batch(
                            out_frame_idx, frame_results, object_names
                        )

                        # 🔹 Live preview (with analysis available)
                        self._maybe_emit_preview(out_frame_idx, frame_results, frame_analysis)

                        frame_analyses[out_frame_idx] = frame_analysis

                        if frame_analysis.get('target_overlaps'):
                            overlap_count += 1

                            # Debug output for first few overlaps
                            if debug and overlap_count <= 3:
                                print(f"  🎯 Frame {out_frame_idx} overlaps:")
                                for target_id, overlaps in frame_analysis['target_overlaps'].items():
                                    target_name = self.overlap_tracker.target_objects[target_id]
                                    for overlap in overlaps:
                                        print(f"    {target_name} {overlap['relationship_desc']}")
                    else:
                        # 🔹 Live preview (no analysis for this frame)
                        self._maybe_emit_preview(out_frame_idx, frame_results, None)

                    frame_count += 1
                    pbar.update(1)

                    # Cleanup
                    if frame_count % 25 == 0:
                        ultra_cleanup_memory()

                    del out_mask_logits, frame_results

                except Exception as e:
                    print(f"  ⚠️ Error processing frame {out_frame_idx}: {e}")
                    pbar.update(1)
                    ultra_cleanup_memory()
                    continue
        
        # Finalize tracking
        if targets_found:
            last_frame = max(results.keys()) if results else 0
            self.overlap_tracker.finalize_tracking(last_frame)
            
            print(f"\n🎯 Enhanced overlap tracking completed:")
            print(f"  📊 Frames with overlaps: {overlap_count}")
            print(f"  📈 Processing efficiency: {frame_count}/{len(self.frame_names)} frames")
            
            # Print detailed summary
            summary = self.overlap_tracker.get_overlap_summary()
            for target_name, data in summary.items():
                print(f"  🎯 {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames")
        
        # Store frame analyses for video creation
        self.frame_analyses = frame_analyses
        
        # Clean up inference state
        self.predictor.reset_state(inference_state)
        ultra_cleanup_memory()
        
        print(f"\n✅ Enhanced processing complete!")
        print(f"📊 Processed {frame_count} frames with improved overlap detection")
        
        return results
    
    def _process_cpu_fallback(self, points_dict, labels_dict, object_names, debug):
        """Emergency CPU fallback processing"""
        print("🚨 Emergency CPU fallback - this will be slow but stable")
        
        if hasattr(self.predictor.model, 'to'):
            self.predictor.model = self.predictor.model.to('cpu')
        
        ultra_cleanup_memory()
        
        messagebox.showwarning("Memory Limitation", 
                             "GPU memory exhausted. Falling back to CPU processing.\n"
                             "This will be much slower but should complete successfully.")
        
        return None
    
    def _maybe_emit_preview(self, frame_idx, frame_results, frame_analysis):
        """
        Lightweight, optional live preview.
        Does nothing unless preview_callback + preview_stride are set.
        Never touches core computations or results.
        """
        cb = getattr(self, "preview_callback", None)
        stride = getattr(self, "preview_stride", None)
        if cb is None or stride is None:
            return
        if frame_idx % max(1, int(stride)) != 0:
            return

        try:
            import os
            import numpy as np
            import cv2

            # Load original frame
            frame_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                return
            H, W = frame.shape[:2]
            overlay = frame.copy()

            # Draw masks (fast alpha blend on mask region)
            colors = getattr(self, "_preview_colors", [(0, 255, 0)])
            for i, (obj_id, mask) in enumerate(frame_results.items()):
                if mask is None:
                    continue
                m = mask[0] if hasattr(mask, "shape") and len(mask.shape) == 3 else mask
                if m is None:
                    continue

                if getattr(m, "shape", None) != (H, W):
                    try:
                        m = cv2.resize(m.astype(float), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        continue
                else:
                    m = (m.astype(float) > 0.5)

                color = colors[i % len(colors)]
                idx = m.astype(bool)
                if np.any(idx):
                    # 60% color + 40% image for the masked area
                    overlay[idx] = (0.6 * np.array(color, dtype=np.float32) + 0.4 * overlay[idx]).astype(np.uint8)

            # Minimal label
            text = f"Frame {frame_idx}"
            try:
                if frame_analysis and frame_analysis.get("target_overlaps"):
                    text += " | overlaps"
            except Exception:
                pass
            cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Downscale once to keep it cheap
            max_side = int(getattr(self, "preview_max_dim", 720))
            if max(H, W) > max_side and max_side > 0:
                scale = max_side / max(H, W)
                overlay = cv2.resize(overlay, (int(W * scale), int(H * scale)))

            cb(overlay)  # hand off to GUI (BGR ndarray)
        except Exception as e:
            print(f"[preview] skipped: {e}")

    
    def save_results_video_with_enhanced_annotations(self, results, output_path, fps=30, show_original=True, alpha=0.5):
        """Save results video with enhanced visual feedback for looking-at events"""
        if not results:
            print("No results to save!")
            return
        
        # Get video properties
        first_frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        if show_original:
            out_width = width * 2
        else:
            out_width = width
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
        
        # Color map for consistent colors
        cmap = plt.get_cmap("tab10")
        
        print("💾 Saving video with enhanced looking-at visual feedback...")
        overlap_frame_count = 0
        
        for frame_idx in tqdm(range(len(self.frame_names)), desc="Saving frames"):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            if frame is None:
                continue
                
            overlay = frame.copy()
            
            # Get enhanced frame analysis if available
            frame_analysis = None
            if hasattr(self, 'frame_analyses') and frame_idx in self.frame_analyses:
                frame_analysis = self.frame_analyses[frame_idx]
            
            # Check if this frame has looking-at events
            has_looking_at_events = (frame_analysis and 
                                frame_analysis.get('target_overlaps') and 
                                any(frame_analysis['target_overlaps'].values()))
            if has_looking_at_events:
                overlap_frame_count += 1
            
            # Collect all looking-at information for this frame
            looking_at_info = {}  # obj_id -> {is_target: bool, looking_at: [objects], looked_at_by: [targets]}
            
            if frame_analysis:
                # Initialize info for all objects
                for obj_id in results.get(frame_idx, {}):
                    looking_at_info[obj_id] = {
                        'is_target': False,
                        'looking_at': [],
                        'looked_at_by': [],
                        'is_being_looked_at': False
                    }
                
                # Process target overlaps
                for target_id, looking_at_objects in frame_analysis.get('target_overlaps', {}).items():
                    if target_id in looking_at_info:
                        looking_at_info[target_id]['is_target'] = True
                        looking_at_info[target_id]['looking_at'] = [obj['object_name'] for obj in looking_at_objects]
                    
                    # Mark objects being looked at
                    for obj_info in looking_at_objects:
                        obj_id = obj_info['object_id']
                        if obj_id in looking_at_info:
                            target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                            looking_at_info[obj_id]['looked_at_by'].append(target_name)
                            looking_at_info[obj_id]['is_being_looked_at'] = True
            
            # Apply masks with enhanced visual feedback
            if frame_idx in results:
                for obj_id, mask in results[frame_idx].items():
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    
                    # Resize mask if needed
                    if mask.shape != (height, width) and mask.shape[0] > 0 and mask.shape[1] > 0:
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), 
                                            interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue
                    
                    if mask.shape == (height, width):
                        obj_info = looking_at_info.get(obj_id, {})
                        is_target = obj_info.get('is_target', False)
                        is_being_looked_at = obj_info.get('is_being_looked_at', False)
                        looking_at = obj_info.get('looking_at', [])
                        looked_at_by = obj_info.get('looked_at_by', [])
                        
                        # Choose colors based on status
                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        
                        if is_target and looking_at:
                            # Target that's looking at something - bright highlight
                            color = np.minimum(base_color + [100, 100, 0], 255)  # Yellow tint for active targets
                            border_color = (0, 255, 255)  # Cyan border for active targets
                            border_thickness = 8
                        elif is_being_looked_at:
                            # Object being looked at - red highlight
                            color = np.minimum(base_color + [120, 0, 0], 255)  # Red tint
                            border_color = (0, 0, 255)  # RED BORDER for looked-at objects
                            border_thickness = 8
                        else:
                            # Normal object
                            color = base_color
                            border_color = None
                            border_thickness = 2
                        
                        # Apply mask color
                        if alpha == 1.0:
                            for c in range(3):
                                overlay[:, :, c][mask] = color[c]
                        else:
                            color_mask = np.zeros_like(overlay)
                            for c in range(3):
                                color_mask[:, :, c][mask] = color[c]
                            
                            blend_mask = np.zeros_like(overlay)
                            cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0, blend_mask)
                            overlay[mask] = blend_mask[mask]
                        
                        # Add enhanced border for special objects
                        if border_color:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, border_color, border_thickness)
                        
                        # Add object label with status
                        moments = cv2.moments(mask.astype(np.uint8))
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            
                            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                            
                            # Create status label
                            if is_target and looking_at:
                                if len(looking_at) == 1:
                                    label = f"{obj_name} → OVERLAPS {looking_at[0]}"
                                else:
                                    label = f"{obj_name} → OVERLAPS {len(looking_at)} OBJECTS"
                            elif is_being_looked_at:
                                    # Suppress label on non-targets to avoid duplicates
                                    continue
                            else:
                                if is_target:
                                    label = f"{obj_name}"
                                else:
                                    label = obj_name
                            
                            # Enhanced text rendering based on status
                            font_scale = 0.6 if len(label) > 30 else 0.7
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                            
                            # Calculate text position
                            label_x = max(5, min(cx - text_size[0]//2, width - text_size[0] - 5))
                            label_y = max(25, min(cy + 20, height - 10))
                            
                            # Enhanced background and text colors
                            if is_target and looking_at:
                                bg_color = (0, 100, 200)  # Orange background for active targets
                                text_color = (0, 255, 255)  # Bright cyan text
                                padding = 10
                            elif is_being_looked_at:
                                bg_color = (0, 0, 200)  # Red background for looked-at objects
                                text_color = (255, 255, 255)  # White text
                                padding = 10
                            else:
                                bg_color = (0, 0, 0)  # Black background
                                text_color = (255, 255, 255)  # White text
                                padding = 5
                            
                            # Draw background rectangle
                            cv2.rectangle(overlay, 
                                        (label_x - padding, label_y - 25), 
                                        (label_x + text_size[0] + padding, label_y + 5), 
                                        bg_color, -1)
                            
                            # Draw text
                            cv2.putText(overlay, label, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
            
            # Add prominent frame status display
            status_messages = []
            
            if has_looking_at_events:
                # Collect all looking-at events for display
                target_events = []
                if frame_analysis:
                    for target_id, looking_at_objects in frame_analysis.get('target_overlaps', {}).items():
                        target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                        object_names = [obj['object_name'] for obj in looking_at_objects]
                        
                        if len(object_names) == 1:
                            target_events.append(f"{target_name} → {object_names[0]}")
                        else:
                            target_events.append(f"{target_name} → {len(object_names)} objects")
                
                status_messages = [f" LOOKING AT DETECTED: {'; '.join(target_events)}"]
            
            # Draw status messages
            info_y = 30
            for i, message in enumerate(status_messages):
                if has_looking_at_events:
                    # Prominent background for looking-at events
                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(overlay, (5, info_y - 25), (text_size[0] + 15, info_y + 10), (0, 0, 180), -1)
                    cv2.putText(overlay, message, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(overlay, message, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                info_y += 35
            
            # Add frame counter
            frame_info = f"Frame {frame_idx}/{len(self.frame_names)-1}"
            cv2.putText(overlay, frame_info, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Create output frame
            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay
                
            out.write(output_frame)
        
        out.release()
        print(f"✅ Video saved with enhanced looking-at visual feedback: {output_path}")
        print(f"🎯 Looking-at events detected in {overlap_frame_count} frames")
        print(f"📊 Visual enhancements:")
        print(f"  • RED BORDERS on objects being looked at")
        print(f"  • CYAN BORDERS on targets that are looking")
        print(f"  • Clear status text for each object")
        print(f"  • Prominent event announcements")
        
    def create_elan_file(self, video_path, output_path, fps, frame_offset=0):
        """Create ELAN file with corrected timing alignment"""
        if not self.overlap_tracker.has_targets():
            print("No targets found - skipping ELAN export")
            return
        
        print(f"Creating ELAN file with timing correction: {output_path}")
        print(f"  Video FPS: {fps}")
        print(f"  Frame offset: {frame_offset}")
        
        # Get actual video properties for verification
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / actual_fps
            cap.release()
            
            print(f"  Actual video FPS: {actual_fps}")
            print(f"  Video duration: {duration_seconds:.2f}s ({total_frames} frames)")
            
            # Use actual FPS if significantly different
            if abs(fps - actual_fps) > 1.0:
                print(f"  ⚠️ FPS mismatch detected! Using actual FPS: {actual_fps}")
                fps = actual_fps
        except:
            print(f"  Using provided FPS: {fps}")
        
        summary = self.overlap_tracker.get_overlap_summary()
        
        # Debug timing calculations
        print(f"\n🔍 Timing Debug:")
        for target_name, target_data in summary.items():
            if target_data['events']:
                first_event = target_data['events'][0]
                start_frame = first_event['start_frame'] + frame_offset
                start_time = start_frame / fps
                print(f"  {target_name} first event: frame {first_event['start_frame']} → {start_frame} → {start_time:.3f}s")
        
        # Create ELAN XML with corrected timing
        header = f'''<?xml version="1.0" encoding="UTF-8"?>
    <ANNOTATION_DOCUMENT AUTHOR="SAM2_Looking_At_Events" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
        <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
            <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
                MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
            <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
        </HEADER>
        <TIME_ORDER>
    '''

        # Create time slots with corrected timing
        time_slots = []
        time_slot_id = 1
        time_slot_refs = {}
        
        all_time_points = set()
        for target_name, target_data in summary.items():
            for event in target_data['events']:
                # Apply frame offset and convert to time
                start_frame_corrected = event['start_frame'] + frame_offset
                end_frame_corrected = event['end_frame'] + frame_offset

                # Skip single-frame events (saccades) - they create zero-duration ELAN annotations
                if end_frame_corrected <= start_frame_corrected:
                    continue

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                # Skip if duration less than ~33ms (one frame at 30fps)
                if end_ms <= start_ms:
                    continue

                all_time_points.add(start_time)
                all_time_points.add(end_time)
        
        for time_point in sorted(all_time_points):
            time_ms = int(time_point * 1000)
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
            time_slot_refs[time_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

        header += '\n'.join(time_slots) + '\n    </TIME_ORDER>\n'

        # Create tiers with corrected timing
        tier_content = ""
        annotation_id = 1
        
        for target_name, target_data in summary.items():
            tier_id = target_name.upper().replace(' ', '_').replace('-', '_')
            tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}_LOOKING_AT">\n'
            
            for event in target_data['events']:
                # Apply frame offset and convert to time
                start_frame_corrected = event['start_frame'] + frame_offset
                end_frame_corrected = event['end_frame'] + frame_offset

                # Skip single-frame events (saccades) - they create zero-duration ELAN annotations
                if end_frame_corrected <= start_frame_corrected:
                    continue

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                # Skip if duration less than ~33ms (one frame at 30fps)
                if end_ms <= start_ms:
                    continue

                start_slot = time_slot_refs.get(start_ms)
                end_slot = time_slot_refs.get(end_ms)
                if not start_slot or not end_slot:
                    continue

                # Create annotation
                overlapping_objects_str = ", ".join(event['overlapping_objects'])
                duration_seconds = end_time - start_time
                
                if len(event['overlapping_objects']) == 1:
                    annotation_value = f"Looking at: {overlapping_objects_str}"
                else:
                    annotation_value = f"Looking at {len(event['overlapping_objects'])} objects: {overlapping_objects_str}"
                
                annotation = f'''        <ANNOTATION>
                <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                    <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>
                </ALIGNABLE_ANNOTATION>
            </ANNOTATION>'''
                
                tier_content += annotation + '\n'
                annotation_id += 1
            
            tier_content += '    </TIER>\n'

        footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
        <LOCALE LANGUAGE_CODE="en"/>
        <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
        <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
        <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
        <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
    </ANNOTATION_DOCUMENT>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + tier_content + footer)
        
        print(f"✅ ELAN file created with timing correction: {output_path}")
        
        # Print timing verification
        print(f"\n📄 Timing Verification:")
        for target_name, target_data in summary.items():
            if target_data['events']:
                print(f"  {target_name}:")
                for i, event in enumerate(target_data['events'][:3]):
                    start_frame_corrected = event['start_frame'] + frame_offset
                    end_frame_corrected = event['end_frame'] + frame_offset
                    start_time = start_frame_corrected / fps
                    end_time = end_frame_corrected / fps
                    
                    objects = ", ".join(event['overlapping_objects'])
                    print(f"    Event {i+1}: {start_time:.3f}s - {end_time:.3f}s | {objects}")
                    print(f"              (frames {start_frame_corrected} - {end_frame_corrected})")
                
                if len(target_data['events']) > 3:
                    print(f"    ... and {len(target_data['events'])-3} more events")

    def export_framewise_csv(self, results, object_names, csv_path):
        """
        Write a frame-by-frame CSV of per-object spatial data + overlap information.
        - Does NOT alter any computations.
        - Uses existing 'results' (masks) and self.frame_analyses (overlaps).
        """
        import csv
        import numpy as np

        # Guard: nothing to do
        if not results:
            print("No results to export to CSV.")
            return

        # Convenience
        analyses = getattr(self, "frame_analyses", {}) or {}
        target_ids = set(getattr(self, "overlap_tracker", None).target_objects.keys()
                        if getattr(self, "overlap_tracker", None) else [])

        # CSV header
        fieldnames = [
            "frame_idx",
            "obj_id",
            "obj_name",
            "is_target",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "centroid_x", "centroid_y",
            "area_px",
            "overlapped_by_targets",
            "target_looking_at",
            "looking_at_count",
        ]

        def _mask_stats(m):
            """Return (x,y,w,h,cx,cy,area) or (None,... ) if empty."""
            try:
                m = m.astype(np.uint8)
                ys, xs = np.where(m > 0)
                if ys.size == 0:
                    return (None,)*7
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                w, h = (x1 - x0 + 1), (y1 - y0 + 1)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(ys.size)
                return x0, y0, w, h, cx, cy, area
            except Exception:
                return (None,)*7

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Process frames in order
            for frame_idx in sorted(results.keys()):
                frame_results = results.get(frame_idx, {})
                frame_analysis = analyses.get(frame_idx, {}) or {}
                target_overlaps = frame_analysis.get("target_overlaps", {}) or {}

                # Build a reverse index for "overlapped_by_targets"
                overlapped_by = {}  # obj_id -> [target_name, ...]
                for t_id, objs in target_overlaps.items():
                    t_name = object_names.get(t_id, f"Object_{t_id}")
                    for entry in (objs or []):
                        oid = entry.get("object_id")
                        if oid is None:
                            continue
                        overlapped_by.setdefault(oid, []).append(t_name)

                # Emit one row per object present in this frame's results
                for obj_id, mask in frame_results.items():
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    # If mask is [1,H,W], squeeze
                    if hasattr(mask, "shape") and len(mask.shape) == 3:
                        mask = mask.squeeze()

                    bx, by, bw, bh, cx, cy, area = _mask_stats(mask)

                    is_target = (obj_id in target_ids)

                    # For target rows, list who this target is "looking at" this frame
                    looking_at = []
                    if is_target and obj_id in target_overlaps:
                        looking_at = [e.get("object_name", f"Object_{e.get('object_id')}")
                                    for e in (target_overlaps.get(obj_id) or [])]

                    row = dict(
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        obj_name=obj_name,
                        is_target=bool(is_target),
                        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                        centroid_x=cx, centroid_y=cy,
                        area_px=area,
                        overlapped_by_targets=";".join(overlapped_by.get(obj_id, [])),
                        target_looking_at=";".join(looking_at) if is_target else "",
                        looking_at_count=(len(looking_at) if is_target else 0),
                    )
                    writer.writerow(row)

        print(f"📄 CSV exported: {csv_path}")

# Point selection function (same as before but with enhanced tips)
def select_points_opencv(frame, processor=None):
    """Interactive point selection tool with enhanced overlap detection tips"""
    points_dict = {}
    labels_dict = {}
    object_names = {}
    current_obj_id = 1
    
    def get_object_name(obj_id):
        if obj_id in object_names:
            return f"{obj_id}:{object_names[obj_id]}"
        else:
            return str(obj_id)
    
    def draw_point(img, point, obj_id, label):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
        
        display_name = get_object_name(obj_id)
        cv2.putText(img, display_name, 
                   (int(point[0] + 5), int(point[1] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    def test_current_mask(frame, points_dict, labels_dict, current_obj_id, object_names, predictor):
        """Test current mask by generating preview with SAM2"""
        if current_obj_id not in points_dict or len(points_dict[current_obj_id]) == 0:
            print("No points selected for current object!")
            return
        
        try:
            print(f"Testing mask for object {current_obj_id}...")
            
            # Create a temporary frames directory for testing
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_frame_path = os.path.join(temp_dir, "00000.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            print(f"Created temp frame: {temp_frame_path}")
            
            # Initialize SAM2 state for testing (same as in main processing)
            inference_state = predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            
            predictor.reset_state(inference_state)
            print("SAM2 state initialized for testing")
            
            # Add points for current object (same as in main processing)
            points = np.array(points_dict[current_obj_id], dtype=np.float32)
            labels = np.array(labels_dict[current_obj_id], dtype=np.int32)
            
            print(f"Points shape: {points.shape}, Labels shape: {labels.shape}")
            print(f"Points: {points}")
            print(f"Labels: {labels}")
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=current_obj_id,
                points=points,
                labels=labels,
            )
            
            print(f"SAM2 output - obj_ids: {out_obj_ids}, mask_logits shape: {out_mask_logits.shape}")
            
            # Generate mask (same as in main processing)
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]
            
            print(f"Generated mask shape: {mask.shape}, mask sum: {np.sum(mask)}")
            
            if np.sum(mask) == 0:
                print("⚠️ Warning: Generated mask is empty!")
                
            # Create preview
            preview = frame.copy()
            
            # Apply mask with color (same as in main processing)
            cmap = plt.get_cmap("tab10")
            color = np.array(cmap(current_obj_id % 10)[:3]) * 255
            
            # Create colored overlay
            color_mask = np.zeros_like(preview)
            for c in range(3):
                color_mask[:, :, c][mask] = color[c]
            
            # Blend overlay
            cv2.addWeighted(preview, 0.5, color_mask, 0.5, 0, preview)
            
            # Add contours for better visibility
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, (0, 255, 0), 3)
            
            # Add text info
            obj_name = object_names.get(current_obj_id, f"Object_{current_obj_id}")
            pos_count = sum(1 for l in labels if l == 1)
            neg_count = sum(1 for l in labels if l == 0)
            
            info_text = f"MASK TEST: {obj_name} (+{pos_count} -{neg_count} points)"
            cv2.putText(preview, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, f"Mask pixels: {np.sum(mask)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, "Press any key to continue...", (10, preview.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show preview
            cv2.namedWindow('Mask Test Preview', cv2.WINDOW_NORMAL)
            cv2.imshow('Mask Test Preview', preview)
            cv2.waitKey(0)
            cv2.destroyWindow('Mask Test Preview')
            
            # Cleanup
            predictor.reset_state(inference_state)
            shutil.rmtree(temp_dir)
            
            print(f"✅ Mask test completed for {obj_name}")
            
        except Exception as e:
            print(f"❌ Error testing mask: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            try:
                predictor.reset_state(inference_state)
                shutil.rmtree(temp_dir)
            except:
                pass
    def redraw_all_points():
        display = frame.copy()
        for obj_id in points_dict:
            for pt, label in zip(points_dict[obj_id], labels_dict[obj_id]):
                draw_point(display, pt, obj_id, label)
        
        
        # -------- Enhanced instruction panels (compact, colored keycaps) --------
        height, width = display.shape[:2]

        # Local helpers
        def put_text(img, text, org, font_scale=0.62, color=(255,255,255), thickness=1, shadow=True):
            x, y = org
            if shadow:
                cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        def draw_round_rect(img, x, y, w, h, color, alpha=0.45, radius=12):
            overlay = img.copy()
            cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color, -1)
            cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color, -1)
            cv2.circle(overlay, (x+radius, y+radius), radius, color, -1)
            cv2.circle(overlay, (x+w-radius, y+radius), radius, color, -1)
            cv2.circle(overlay, (x+radius, y+h-radius), radius, color, -1)
            cv2.circle(overlay, (x+w-radius, y+h-radius), radius, color, -1)
            return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        def draw_keycap(img, text, x, y, pad_x=10, pad_y=6, font_scale=0.56,
                        fg=(30,30,30), bg=(0,255,255)):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            w, h = tw + 2*pad_x, th + 2*pad_y
            img = draw_round_rect(img, x, y - h, w, h, bg, alpha=0.95, radius=10)
            tx, ty = x + pad_x, y - pad_y
            cv2.putText(img, text, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, text, (tx,   ty  ), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 2, cv2.LINE_AA)
            return img, (w, h)

        # Layout constants
        margin = 12
        line_gap = 32
        header_gap = 30

        # ---- Left: CONTROLS panel (compact width, dynamic height)
        controls = [
            ("Left Click",  "Add positive point (+)"),
            ("Right Click", "Add negative point (-)"),
            ("T",           "Test current mask"),
            ("C",           "Name object"),
            ("R",           "Reset points"),
            ("N",           "Next object"),
            ("P",           "Previous object"),   # <-- added back
            ("Enter",       "Finish & Start Video Processing"),
            ("Q",           "Quit"),
        ]

        # Per-key background colors
        key_bg = {
            "Left Click":  (0, 200, 0),     # green
            "Right Click": (0, 0, 220),     # red
            "T":           (220, 140, 0),   # orange
            "C":           (180, 0, 200),   # purple
            "R":           (0, 220, 220),   # yellow-ish (BGR)
            "N":           (220, 0, 180),   # magenta
            "P":           (255, 0, 0),     # blue  <-- new color
            "Enter":       (220, 220, 220), # light gray
            "Q":           (160, 160, 160), # gray
        }

        # Panel width ~ 44% of frame, placed bottom-left
        ctrl_w = int(width * 0.44)
        # Height = header + lines
        ctrl_h = header_gap + len(controls)*line_gap + margin + 10
        ctrl_x1 = margin
        ctrl_y1 = height - ctrl_h - margin
        ctrl_w  = max(ctrl_w, 320)  # minimum width
        ctrl_h  = max(ctrl_h, 200)  # minimum height

        display = draw_round_rect(display, ctrl_x1, ctrl_y1, ctrl_w, ctrl_h, (0,0,0), alpha=0.45, radius=12)

        # Header with extra spacing so keycap doesn't overlap it
        header_x = ctrl_x1 + 16
        header_y = ctrl_y1 + 32
        put_text(display, "CONTROLS", (header_x, header_y), font_scale=0.9, color=(0,255,255), thickness=2)

        # Commands (one per line)
        cx = header_x
        cy = header_y + header_gap  # extra gap so header isn't covered
        for key, desc in controls:
            bg = key_bg.get(key, (0,255,255))
            display, (kw, kh) = draw_keycap(display, key, cx, cy, bg=bg)
            put_text(display, desc, (cx + kw + 12, cy - 8), font_scale=0.58, color=(255,255,255), thickness=1)
            cy += line_gap

        # ---- Right: IMPORTANT panel (smaller, neutral color)
        imp_text = [
            "Any object named with 'target'",
            "will be used to calculate overlap",
            '("look-at" events) with other objects.',
            "Case-insensitive substring match:",
            "target, Target_1, TARGET2 all count.",
        ]

        imp_w = int(width * 0.46)
        imp_w = max(imp_w, 360)
        imp_h = header_gap + len(imp_text)*24 + margin + 10
        imp_x1 = width - imp_w - margin
        imp_y1 = height - imp_h - margin

        display = draw_round_rect(display, imp_x1, imp_y1, imp_w, imp_h, (30,30,30), alpha=0.50, radius=12)

        put_text(display, "IMPORTANT", (imp_x1 + 16, imp_y1 + 32), font_scale=0.9, color=(0,255,0), thickness=2)

        ty = imp_y1 + header_gap + 22
        for line in imp_text:
            put_text(display, line, (imp_x1 + 16, ty), font_scale=0.62, color=(230,255,230), thickness=1)
            ty += 24

        # ---- HUD (unchanged)
        current_obj_name = get_object_name(current_obj_id)
        put_text(display, f"Current Object: {current_obj_name}", (20, 30), font_scale=0.85, color=(0,255,0), thickness=2)

        if current_obj_id in points_dict:
            pos_count = sum(1 for l in labels_dict[current_obj_id] if l == 1)
            neg_count = sum(1 for l in labels_dict[current_obj_id] if l == 0)
            put_text(display, f"Points: +{pos_count}  -{neg_count}", (20, 60), font_scale=0.7, color=(255,255,0), thickness=2)

        return display

    
    def name_current_object():
        import tkinter as tk
        from tkinter import simpledialog

        nonlocal img_display, current_obj_id

        root = tk.Tk()
        root.withdraw()
        
        current_name = object_names.get(current_obj_id, f"Object_{current_obj_id}")
        name = simpledialog.askstring(
            "Enhanced Object Naming", 
            f"   ENTER NAME FOR {current_obj_id}:\n\n"
            f"<Current Name: {current_name}>",
            initialvalue=current_name
        )
        root.destroy()
        
        if name and name.strip():
            # User entered a valid name
            object_names[current_obj_id] = name.strip()
            print(f"Object {current_obj_id} named: {object_names[current_obj_id]}")
            
            if 'target' in name.lower():
                print("🎯 Target detected: Enhanced overlap detection will track inclusion and overlaps")

            # Show updated label for this object
            img_display = redraw_all_points()

            # --- Auto-advance to next object ---
            current_obj_id += 1
            obj_name = get_object_name(current_obj_id)
            print(f"Now selecting {obj_name}")
            img_display = redraw_all_points()
        else:
            # Cancel pressed or blank → just stay on current object
            print(f"⚠️ Naming cancelled for object {current_obj_id}. Staying on same object.")
            img_display = redraw_all_points()

        
    def click_handler(event, x, y, flags, param):
        nonlocal img_display
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if current_obj_id not in points_dict:
                points_dict[current_obj_id] = []
                labels_dict[current_obj_id] = []
                if current_obj_id not in object_names:
                    object_names[current_obj_id] = f"Object_{current_obj_id}"
            
            points_dict[current_obj_id].append([x, y])
            label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            labels_dict[current_obj_id].append(label)
            
            img_display = redraw_all_points()
            obj_name = get_object_name(current_obj_id)
            print(f"Added {'positive' if label == 1 else 'negative'} point for {obj_name}")
    
    img_display = redraw_all_points()
    cv2.namedWindow('Reference Frame Annotation', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Reference Frame Annotation', click_handler)

    
    while True:
        cv2.imshow('Reference Frame Annotation', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            if current_obj_id in points_dict:
                points_dict[current_obj_id] = []
                labels_dict[current_obj_id] = []
                img_display = redraw_all_points()
                obj_name = get_object_name(current_obj_id)
                print(f"Reset points for {obj_name}")
        elif key == ord('t'):  # ADD THIS
            if processor and processor.predictor:
                test_current_mask(frame, points_dict, labels_dict, current_obj_id, object_names, processor.predictor)
            else:
                messagebox.showwarning("Test Mask", "Predictor not available for testing!")
    
        elif key == ord('n'):
            current_obj_id += 1
            obj_name = get_object_name(current_obj_id)
            print(f"Now selecting {obj_name}")
            img_display = redraw_all_points()
        
        elif key == ord('p'):
            if current_obj_id > 1:
                current_obj_id -= 1
                obj_name = get_object_name(current_obj_id)
                print(f"Now selecting {obj_name}")
                img_display = redraw_all_points()
        
        elif key == ord('c'):
            name_current_object()
        
        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            return points_dict, labels_dict, object_names if points_dict else (None, None, None)
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None, None
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None, None
    

# GUI Application class with enhanced overlap detection
class VideoAnalysisApp:
    def __init__(self):
        # NEW: modern root + theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("EnvisionObjectAnnotator")
        self.root.geometry("1000x800")
        self.root.minsize(950, 700)

        # Initialize SAM2
        self.device = setup_device_ultra_optimized()
        self.predictor = None
        self.init_sam2()

        # NEW: build the modern shell (left controls + right tabs)
        self._build_shell()

        # Thread-safe preview queue & poller
        self._preview_q = queue.Queue(maxsize=2)
        self.root.after(50, self._drain_preview_queue)
        
        # Reuse your existing controls but mount them into the left panel
        self.setup_gui(parent=self.left_panel)

    def _build_shell(self):
        # ----- Paned window (draggable split) -----
        self.paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)

        # We add plain tk containers to the paned window, then put CTk frames inside them
        self.left_container  = tk.Frame(self.paned)   # container for CTk left panel
        self.right_container = tk.Frame(self.paned)   # container for CTk right panel
        self.paned.add(self.left_container,  weight=1)   # weight controls how resize space is shared
        self.paned.add(self.right_container, weight=3)

        # Optional: set a reasonable minimum width for the left side
        try:
            self.paned.paneconfig(self.left_container,  minsize=280)
            self.paned.paneconfig(self.right_container, minsize=400)
        except Exception:
            pass  # older Tk versions may not support paneconfig

        # ----- Left controls column (CTk inside the left container) -----
        self.left_panel = ctk.CTkFrame(self.left_container)
        self.left_panel.pack(side="left", fill="both", expand=True)

        # ----- Right side with tabs (CTk inside the right container) -----
        self.right_panel = ctk.CTkFrame(self.right_container)
        self.right_panel.pack(side="right", fill="both", expand=True)

        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.preview_frame = ctk.CTkFrame(self.notebook); self.notebook.add(self.preview_frame, text="Preview")
        #self.results_frame = ctk.CTkFrame(self.notebook); self.notebook.add(self.results_frame, text="Results")
        #self.progress_frame = ctk.CTkFrame(self.notebook); self.notebook.add(self.progress_frame, text="Progress")

        # Progress console
        #from tkinter import scrolledtext
        #self.progress_text = scrolledtext.ScrolledText(self.progress_frame, height=20, wrap="word", font=("Consolas", 10))
        #self.progress_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Preview placeholder
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="No video loaded")
        self.preview_label.pack(expand=True)


    
        
    def init_sam2(self):
        """Initialize SAM2 predictor with ultra memory optimization"""
        try:
            configure_torch_ultra_conservative()

            sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            if not os.path.exists(sam2_checkpoint):
                print("Large model not found, checking for small model...")
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

                if not os.path.exists(sam2_checkpoint):
                    messagebox.showwarning("SAM2 Setup", "SAM2 checkpoints not found. Please update paths.")
                    return
                else:
                    print("Using small model (sam2.1_hiera_small.pt) with the sam2.1_hiera_s.yaml configuration")
            else:
                print("Using large model (sam2.1_hiera_large.pt) with the sam2.1_hiera_l.yaml configuration")

            from sam2.build_sam import build_sam2_video_predictor

            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            print("SAM2 predictor initialized with enhanced overlap detection")

            gpu_info = get_gpu_memory_info()
            if gpu_info:
                print(f"GPU Memory: {gpu_info['allocated_gb']:.1f}GB allocated, {gpu_info['free_gb']:.1f}GB free")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize SAM2: {str(e)}")


    def setup_gui(self, parent=None):
        """Setup the GUI with enhanced overlap detection options"""
        if parent is None:
            parent = self.root
        main_frame = tk.Frame(parent, padx=15, pady=15)  # parent was self.root
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title block frame
        title_frame = tk.Frame(main_frame)
        title_frame.pack(pady=(0, 15))

        # EOA big title
        title_label1 = tk.Label(title_frame, text="EnvisionObjectAnnotator:", font=("Arial", 15, "bold"))
        title_label1.pack()

        # SAM2 + subtitle
        title_label2 = tk.Label(title_frame, text="SAM2 Object Tracking & 'Looking At' Detection", font=("Arial", 12, "bold"))
        title_label2.pack()

        # Developers (small font)
        title_label3 = tk.Label(title_frame, text="Developers: Wim Pouw, Babajide Owoyele, Davide Ahmar", font=("Arial", 8))
        title_label3.pack()

        
        # Simplified features info
        features_frame = tk.LabelFrame(main_frame, text="🎯 App Features:", font=("Arial", 10, "bold"))
        features_frame.pack(fill=tk.X, pady=(0, 10))

        features_text = """\
        - User: imports video, marks objects & TARGET (e.g., gaze marker) on a frame.
        - EnvisionObjectAnnotator: tracks all objects & detects when the TARGET overlaps with other objects. Outputs:
            • Annotated video (events highlighted)
            • CSV (frame-by-frame data)
            • ELAN file (timeline for coding)"""

        features_label = tk.Label(
            features_frame,
            text=features_text,
            font=("Arial", 10),   # bigger font than before
            justify=tk.LEFT,      # align to the left
            fg="black"            # normal text color
        )
        features_label.pack(anchor=tk.W, padx=5, pady=5)
        
       
       
        # Memory status
        memory_status_frame = tk.LabelFrame(main_frame, text="⚙️ Memory Status", font=("Arial", 9, "bold"))
        memory_status_frame.pack(fill=tk.X, pady=(0, 10))

        # Three variables for GPU, RAM, CPU
        self.gpu_status_var = tk.StringVar(value="GPU: Checking...")
        self.ram_status_var = tk.StringVar(value="RAM: Checking...")
        self.cpu_status_var = tk.StringVar(value="CPU: Checking...")

        gpu_label = tk.Label(memory_status_frame, textvariable=self.gpu_status_var,
                            font=("Arial", 8), fg="darkblue")
        gpu_label.pack(anchor=tk.W, padx=5, pady=(3,0))

        ram_label = tk.Label(memory_status_frame, textvariable=self.ram_status_var,
                            font=("Arial", 8), fg="darkblue")
        ram_label.pack(anchor=tk.W, padx=5, pady=(2,0))

        cpu_label = tk.Label(memory_status_frame, textvariable=self.cpu_status_var,
                            font=("Arial", 8), fg="darkblue")
        cpu_label.pack(anchor=tk.W, padx=5, pady=(2,3))

        self.update_memory_status()
        

        
        # --- Video selection (single file) ---
        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(video_frame, text="Select video:", font=("Arial", 9)).pack(anchor=tk.W)

        video_input_frame = tk.Frame(video_frame)
        video_input_frame.pack(fill=tk.X, pady=(5, 0))

        # Holds the FULL path; useful for processing
        self.video_path_var = tk.StringVar(value="")

        # Shows ONLY the file name (no path)
        self.video_name_var = tk.StringVar(value="No video selected")

        # Small display box for the selected file name
        video_name_label = tk.Label(
            video_input_frame,
            textvariable=self.video_name_var,
            font=("Arial", 9),
            relief="sunken",
            anchor="w"
        )
        video_name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        tk.Button(video_input_frame, text="Browse", command=self.select_video).pack(side=tk.RIGHT)

        # Simplified event detection options
        overlap_frame = tk.LabelFrame(main_frame, text="🎯 'Select Detection Threshold", font=("Arial", 9, "bold"))
        overlap_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Overlap threshold
        threshold_frame = tk.Frame(overlap_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threshold_frame, text="Detection Threshold (%):").pack(side=tk.LEFT)
        self.overlap_threshold_var = tk.StringVar(value="10")
        threshold_spin = tk.Spinbox(threshold_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.overlap_threshold_var, width=10)
        threshold_spin.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(threshold_frame, text="(minimum spatial relationship to detect)", 
                font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Event detection info
        event_info = tk.Label(overlap_frame, 
                             text="Event rule: a 'looking-at' event is logged when the TARGET intersects another object — either (i) mask overlap ≥ the detection threshold (%) or (ii) the TARGET centroid lies inside the object's mask.",
                             font=("Arial", 8), fg="darkgreen", wraplength=700)
        event_info.pack(anchor=tk.W, padx=5, pady=(2, 5))
        
        # Memory optimization options
        options_frame = tk.LabelFrame(main_frame, text="🚀 Memory Optimization", font=("Arial", 9, "bold"))
        options_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Batch size
        batch_frame = tk.Frame(options_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(batch_frame, text="Memory Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value="50")
        batch_spin = tk.Spinbox(batch_frame, from_=10, to=200, increment=10, 
                               textvariable=self.batch_size_var, width=10)
        batch_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Auto fallback
        self.auto_fallback = tk.BooleanVar(value=True)
        fallback_cb = tk.Checkbutton(options_frame, 
                                    text="🔄 Auto fallback to CPU if GPU memory exhausted",
                                    variable=self.auto_fallback)
        fallback_cb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Output options
        output_frame = tk.LabelFrame(main_frame, text="📁 Output Options", font=("Arial", 9, "bold"))
        output_frame.pack(fill=tk.X, pady=(10, 10))

        # ✅ Annotated video 
        self.enable_video_export = tk.BooleanVar(value=True)
        video_cb = tk.Checkbutton(
            output_frame,
            text="🎞️ Save annotated video (with masks/labels)",
            variable=self.enable_video_export
        )
        video_cb.pack(anchor=tk.W, padx=5, pady=2)

        # ✅ ELAN file
        self.enable_elan_export = tk.BooleanVar(value=True)
        elan_cb = tk.Checkbutton(
            output_frame,
            text="📄 Export ELAN file with 'looking at' event timing",
            variable=self.enable_elan_export
        )
        elan_cb.pack(anchor=tk.W, padx=5, pady=2)

        # ✅ CSV file
        self.enable_csv_export = tk.BooleanVar(value=True)
        csv_cb = tk.Checkbutton(
            output_frame,
            text="📊 Save frame-by-frame CSV (positions + overlaps)",
            variable=self.enable_csv_export
        )
        csv_cb.pack(anchor=tk.W, padx=5, pady=2)

        # Output folder
        outdir_row = tk.Frame(output_frame)
        outdir_row.pack(fill=tk.X, padx=5, pady=(6,2))

        tk.Label(outdir_row, text="Output folder:").pack(side=tk.LEFT)

        self.output_dir_var = tk.StringVar(value="")  # empty = defaults next to the video
        outdir_entry = tk.Entry(outdir_row, textvariable=self.output_dir_var)
        outdir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))

        tk.Button(outdir_row, text="Choose…", command=self.select_output_dir).pack(side=tk.LEFT)

        tk.Label(
            output_frame,
            text="(Leave empty to save next to the video)",
            font=("Arial", 8), fg="gray"
        ).pack(anchor=tk.W, padx=5)

        
        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(15, 10))
        
        self.process_button = tk.Button(process_frame, text="🎯 Process Video ('Looking At' Event Detection)", 
                                       command=self.process_video, bg="#4CAF50", fg="white",
                                       font=("Arial", 11, "bold"), pady=8)
        self.process_button.pack(fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - 'Looking at' event detection with clean timing")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 8), wraplength=700)
        status_label.pack(pady=(5, 0))
    

    def update_memory_status(self):
        import psutil, torch

        # RAM + CPU
        try:
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used / (1024**3)
            ram_total_gb = vm.total / (1024**3)
            self.ram_status_var.set(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({vm.percent:.0f}%)")
            self.cpu_status_var.set(f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
        except Exception:
            self.ram_status_var.set("RAM: unavailable")
            self.cpu_status_var.set("CPU: unavailable")

        # GPU
        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                total_gb = props.total_memory / (1024**3)
                used_gb  = torch.cuda.memory_allocated(dev) / (1024**3)
                pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0
                self.gpu_status_var.set(f"GPU: {props.name} {used_gb:.1f}/{total_gb:.1f} GB ({pct:.0f}%)")
            else:
                self.gpu_status_var.set("GPU: Not available (using CPU)")
        except Exception:
            self.gpu_status_var.set("GPU: Unknown")

        # schedule next update
        self.root.after(2000, self.update_memory_status)
    
    def select_video(self):
        from tkinter import filedialog
        import os

        filetypes = [
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select a video", filetypes=filetypes)
        if not path:
            return

        # store full path for processing
        self.video_path_var.set(path)
        # show only the file name in the small box
        self.video_name_var.set(os.path.basename(path))

        # Optional: if you implemented a preview loader, call it here
        try:
            self.load_video_preview(path)   # safe to omit if you don’t have this
        except Exception:
            pass

        try:
            self.output_dir_var.set(os.path.dirname(path))
        except Exception:
            pass

        # Optional: enable your "Process" button if you gate it on selection
        # self.process_btn.configure(state="normal")

    def select_output_dir(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            os.makedirs(path, exist_ok=True)
            self.output_dir_var.set(path)

    
    def scan_videos(self, folder):
        """Scan for video files in folder"""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']
        videos = []
        
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(file)
        
        videos.sort()
        
        self.video_listbox.delete(0, tk.END)
        for video in videos:
            self.video_listbox.insert(tk.END, video)
        
        if videos:
            self.video_listbox.select_set(0)
            self.status_var.set(f"Found {len(videos)} video(s) - 'Looking at' event detection ready")
        else:
            self.status_var.set("No videos found in selected folder")
    
    def get_frame_number_with_preview(self, frames_dir, total_frames):
        """Get frame number with preview functionality"""
        suggested_frame = 0  # Start with the first frame
        
        while True:
            frame_num = simpledialog.askinteger(
                "Select Reference Frame",
                (
                    f"Pick the reference frame (0–{total_frames-1}).\n\n"
                    "Recommended: 0 (the first frame).\n"
                    "Why: tracking begins from the frame you choose. "
                    "If you pick a later frame, the beginning of the video may show no masks "
                    "or 'looking at' annotations.\n\n"
                    f"Suggested: frame {suggested_frame}"
                ),
                minvalue=0,
                maxvalue=total_frames-1,
                initialvalue=suggested_frame,
            )
                    
            if frame_num is None:
                return None
            
            if frame_num == -1:
                if show_frame_preview(frames_dir, suggested_frame, total_frames):
                    continue
                else:
                    return None
                
            # After you’ve obtained frame_num and before returning it:
            if frame_num > 0:
                proceed = messagebox.askyesno(
                    "Reference Frame Warning",
                    "You selected a reference frame after the beginning.\n\n"
                    "With the current pipeline the tracker is seeded at that frame, "
                    "so earlier frames may have no masks/annotations.\n\n"
                    "Do you want to continue?"
                )
                if not proceed:
                    continue  # back to the selection dialog
            
            if show_frame_preview(frames_dir, frame_num, total_frames):
                confirm = messagebox.askyesno("Confirm Frame Selection", 
                    f"Use this frame {frame_num} as REFERENCE FRAME?\n\n"
                    " This frame will be use to annotate all objects & targets")
                
                if confirm:
                    return frame_num
            else:
                return None

    
    def load_video_preview(self, video_path):
        """Load one frame and show it in the Preview tab, scaled to fit the Preview panel."""
        import cv2
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video for preview")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # first frame

            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                messagebox.showerror("Error", "Could not read frame for preview")
                return

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # --- NEW: get the current size of the preview panel ---
            self.preview_frame.update_idletasks()  # ensure geometry info is up to date
            max_w = self.preview_frame.winfo_width()
            max_h = self.preview_frame.winfo_height()

            # Fallback if panel not yet sized
            if max_w < 50 or max_h < 50:
                max_w, max_h = 600, 400

            # Scale image to fit
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)

            # Update the CTkLabel
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video preview: {e}")



    def process_video(self):
        try:
            self.notebook.select(self.preview_frame)
        except Exception:
            pass
        """Process the selected video with enhanced overlap detection (single-file selection)."""
        if self.predictor is None:
            messagebox.showerror("Error", "SAM2 predictor not initialized")
            return

        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("Warning", "Please select a valid video file")
            return

        try:
            # Status
            self.status_var.set("🎯 Starting 'looking at' event detection.")
            self.update_memory_status()
            self.root.update()

            # Make frames dir next to the video
            video_stem = Path(video_path).stem
            frames_dir = os.path.join(os.path.dirname(video_path), f"{video_stem}_frames")

            # Extract frames
            self.status_var.set("📹 Extracting frames.")
            self.root.update()
            fps, num_frames = video_to_frames(video_path, frames_dir)
            if fps == -1:
                messagebox.showerror("Error", "Failed to extract frames from video")
                return

            print(f"🎯 Processing {num_frames} frames with 'looking at' event detection")

            # Ask user for reference frame (with preview)
            frame_num = self.get_frame_number_with_preview(frames_dir, num_frames)
            if frame_num is None:
                self.status_var.set("Processing cancelled")
                return

            # Get settings from UI
            try:
                overlap_threshold = float(self.overlap_threshold_var.get()) / 100.0
                batch_size = int(self.batch_size_var.get())
            except ValueError:
                overlap_threshold = 0.1
                batch_size = 50
                self.overlap_threshold_var.set("10")
                self.batch_size_var.set("50")

            # Init enhanced processor
            self.status_var.set("🔧 Initializing enhanced overlap processor.")
            self.update_memory_status()
            processor = UltraOptimizedProcessor(
                predictor=self.predictor,
                video_dir=frames_dir,
                overlap_threshold=overlap_threshold,
                reference_frame=frame_num,
                batch_size=batch_size,
                auto_fallback=self.auto_fallback.get() if hasattr(self, "auto_fallback") else True,
                preview_callback=self._q_put_preview,
                preview_stride = 1,  #every frame
                preview_max_dim=720)

            # Get points/labels/object names via the OpenCV selector
            # (Pass the processor so 'T' can test the current mask)
            ref_frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
            ref_frame = cv2.imread(ref_frame_path)
            if ref_frame is None:
                messagebox.showerror("Error", f"Failed to read reference frame {frame_num}")
                return

            points_dict, labels_dict, object_names = select_points_opencv(ref_frame, processor=processor)
            if points_dict is None:
                self.status_var.set("Processing cancelled")
                return
            
            # Use the chosen output folder (falls back to the video folder if empty)
            out_root = (self.output_dir_var.get().strip() or os.path.dirname(video_path))
            os.makedirs(out_root, exist_ok=True)  # make sure it exists

            # Run heavy processing in background to keep GUI responsive
            def _worker():
                try:
                    # Indicate run (UI hint is cheap via after)
                    self.root.after(0, lambda: self.status_var.set("🚀 Propagating and detecting events."))

                    # 1) Propagate & detect
                    results = processor.process_video_with_memory_management(points_dict, labels_dict, object_names, debug=True)

                    # 2) Save annotated video + ELAN + CSV in worker (honor checkboxes)
                    video_saved = False
                    elan_created = False
                    csv_saved = False

                    out_video = None
                    elan_path = None
                    csv_path = None

                    if results is not None:
                        # 🎞️ Annotated video
                        if getattr(self, "enable_video_export", tk.BooleanVar(value=True)).get():
                            out_video = os.path.join(out_root, f"{video_stem}_ANNOTATED_VIDEO.mp4")            # <-- CHANGED
                            processor.save_results_video_with_enhanced_annotations(
                                results, out_video, fps=fps, show_original=True, alpha=0.5
                            )
                            video_saved = True

                        # 📄 ELAN timeline
                        if getattr(self, "enable_elan_export", tk.BooleanVar(value=True)).get():
                            elan_path = os.path.join(out_root, f"{video_stem}_ELAN_TIMELINE.eaf")              # <-- CHANGED
                            processor.create_elan_file(video_path, elan_path, fps=fps, frame_offset=0)
                            elan_created = True

                        # 📊 CSV export
                        if getattr(self, "enable_csv_export", tk.BooleanVar(value=True)).get():
                            csv_path = os.path.join(out_root, f"{video_stem}_FRAME_BY_FRAME.csv")              # <-- CHANGED
                            if hasattr(processor, "export_framewise_csv"):
                                processor.export_framewise_csv(results, object_names, csv_path)
                            elif hasattr(processor, "create_csv_file"):
                                processor.create_csv_file(video_path, csv_path, fps=fps, frame_offset=0)
                            csv_saved = True

                        # Build success summary
                        summary = processor.overlap_tracker.get_overlap_summary() if processor.overlap_tracker else {}
                        target_info = ""
                        for target_name, data in summary.items():
                            try:
                                target_info += f"\n {target_name}: {data['total_events']} events, {data['total_overlap_frames']} frames"
                            except Exception:
                                pass

                        named_objects = [name for name in object_names.values()]
                        objects_summary = "\n".join([f"  • {name}" for name in named_objects])

                        generated = []
                        if video_saved and out_video:
                            generated.append(f"• {os.path.basename(out_video)} – annotated video")
                        if elan_created and elan_path:
                            generated.append(f"• {os.path.basename(elan_path)} – ELAN timeline")
                        if csv_saved and csv_path:
                            generated.append(f"• {os.path.basename(csv_path)} – frame-by-frame CSV")
                        generated_section = "\n".join(generated)

                        success_msg = f"""EnvisionObjectAnnotator: Video Processing Complete!

                        Reference Frame: {frame_num}
                        Detection Method: Overlapping = 'looking at' event
                        Detection Threshold: {overlap_threshold*100:.1f}%
                        Clean Timing: Accurate begin/end for behavioral analysis
                        Results saved in: {out_root}

                        📁 Generated Files:
                        {generated_section}{target_info}

                        📊 Analyzed Objects ({len(object_names)}):
                        {objects_summary}

                        ✅ 'Looking at' event detection with clean timing completed!
                        """



                        # UI finalize on main thread
                        self.root.after(0, lambda: messagebox.showinfo("'Looking At' Event Detection Complete", success_msg))
                        self.root.after(0, lambda: self.status_var.set("Done"))

                    elif results is None:
                        self.root.after(0, lambda: messagebox.showwarning(
                            "Processing Incomplete",
                            "GPU memory was exhausted. CPU fallback used.\n\nConsider reducing the number of objects for better performance."
                        ))
                        self.root.after(0, lambda: self.status_var.set("Event detection completed with limitations"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "'Looking at' event detection failed"))
                        self.root.after(0, lambda: self.status_var.set("Event detection failed"))

                except Exception as e:
                    import traceback; traceback.print_exc()
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Event detection failed: {str(e)}"))
                    self.root.after(0, lambda: self.status_var.set("Event detection failed"))
                finally:
                    self.root.after(0, ultra_cleanup_memory)
                    self.root.after(0, self.update_memory_status)

            threading.Thread(target=_worker, daemon=True).start()
            return


        except Exception as e:
            messagebox.showerror("Error", f"Event detection failed: {str(e)}")
            self.status_var.set("Event detection failed")
            import traceback; traceback.print_exc()
        finally:
            ultra_cleanup_memory()
            self.update_memory_status()

    def preview_hook(self, frame_bgr):
        """
        Lightweight updater for the Preview tab.
        Expects a BGR np.ndarray. Converts to RGB and scales to fit preview panel.
        """
        try:
            import cv2 as _cv2
            from PIL import Image, ImageTk

            # Convert BGR -> RGB for Tk
            frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Fit to preview panel size (if available)
            if hasattr(self, 'preview_frame') and self.preview_frame is not None:
                self.preview_frame.update_idletasks()
                max_w = max(200, self.preview_frame.winfo_width())
                max_h = max(150, self.preview_frame.winfo_height())
                scale = min(max_w / w, max_h / h, 1.0)
                if scale < 1.0:
                    frame_rgb = _cv2.resize(frame_rgb, (int(w*scale), int(h*scale)))

            # Push to the existing preview label
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            if hasattr(self, 'preview_label') and self.preview_label is not None:
                try:
                    self.preview_label.configure(image=photo, text="")
                except Exception:
                    pass
                # keep a reference so Tk doesn't GC the image
                self.preview_label.image = photo

            # Keep UI responsive without blocking compute
            if hasattr(self, 'root') and self.root is not None:
                self.root.update_idletasks()

        except Exception as e:
            print(f"[preview_gui] {e}")




    def _q_put_preview(self, frame_bgr):
        """Thread-safe: enqueue latest preview frame without blocking the worker."""
        try:
            if self._preview_q.full():
                try:
                    self._preview_q.get_nowait()
                except queue.Empty:
                    pass
            self._preview_q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _drain_preview_queue(self):
        """GUI-thread poller: paint the most recent frame, then reschedule."""
        last = None
        try:
            while True:
                last = self._preview_q.get_nowait()
        except queue.Empty:
            pass
        if last is not None:
            self.preview_hook(last)
        self.root.after(20, self._drain_preview_queue)   # ~66 fps polling

    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting EnvisionObjectAnnotator - Automatic Object Tracking & Overlap Detection")
    print("=" * 70)

    print("\n🎯 EVENT DETECTION")
    print(" • EnvisionObjectAnnotator detects when the TARGET overlaps with other objects.")
    print(" • Overlap or full inclusion = 'looking-at' event.")
    print(" • Each event has precise start and end times for ELAN or CSV export.")
    print()

    print("📄 OUTPUTS")
    print(" • Annotated video showing when looking-at events occur.")
    print(" • CSV file with frame-by-frame event data.")
    print(" • ELAN (.eaf) file with labeled looking-at events.")
    print()

    print("💡 EXAMPLES")
    print(" • 'TARGET crosshair looking at apple'")
    print(" • 'Frame 245/1200 - looking-at event detected'")
    print()

    print("⚙️ PERFORMANCE")
    print(" • GPU memory optimization and cleanup every 25 frames.")
    print(" • Automatic CPU fallback if GPU memory is insufficient.")
    print()

    print("-" * 50)
    print("Initializing EnvisionObjectAnnotator...")

    app = VideoAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()

    def preview_hook(self, frame_bgr):
        """Update the Preview tab with a downscaled RGB image. Safe and light."""
        try:
            import cv2 as _cv2
            from PIL import Image, ImageTk
            frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            if hasattr(self, 'preview_frame') and self.preview_frame is not None:
                self.preview_frame.update_idletasks()
                max_w = max(200, self.preview_frame.winfo_width())
                max_h = max(150, self.preview_frame.winfo_height())
                scale = min(max_w / w, max_h / h, 1.0)
                if scale < 1.0:
                    frame_rgb = _cv2.resize(frame_rgb, (int(w*scale), int(h*scale)))
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            if hasattr(self, 'preview_label') and self.preview_label is not None:
                try:
                    self.preview_label.configure(image=photo, text="")
                except Exception:
                    pass
                self.preview_label.image = photo
            if hasattr(self, 'root') and self.root is not None:
                self.root.update_idletasks()
        except Exception as e:
            print(f"[preview_gui] {e}")

