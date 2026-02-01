"""
Frame analysis utilities for suggesting optimal annotation frames.
Implements hybrid approach: basic metrics + optional DINOv2 enhancement.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate frame sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def calculate_brightness_score(image: np.ndarray) -> float:
    """Calculate brightness score (prefer mid-range, penalize too dark/bright)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    # Score: 1.0 at 127, decreasing toward 0/255
    score = 1.0 - abs(brightness - 127.0) / 127.0
    return float(score)


def calculate_edge_density(image: np.ndarray) -> float:
    """Calculate edge density (more edges = more detail)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0
    return float(edge_density)


def calculate_color_variance(image: np.ndarray) -> float:
    """Calculate color variance (more variance = more diverse content)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_std = hsv[:, :, 0].std()
    s_std = hsv[:, :, 1].std()
    variance = (h_std + s_std) / 2.0
    return float(variance)


def calculate_basic_score(image_path: Path) -> Dict[str, float]:
    """Calculate basic quality scores for a frame."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "sharpness": 0.0,
            "brightness": 0.0,
            "edge_density": 0.0,
            "color_variance": 0.0,
            "combined": 0.0,
        }

    sharpness = calculate_sharpness(image)
    brightness = calculate_brightness_score(image)
    edge_density = calculate_edge_density(image)
    color_variance = calculate_color_variance(image)

    # Combined score: weighted average
    combined = (
        0.4 * min(sharpness / 100.0, 1.0) +  # Normalize sharpness
        0.2 * brightness +
        0.2 * edge_density +
        0.2 * min(color_variance / 50.0, 1.0)  # Normalize variance
    )

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "edge_density": edge_density,
        "color_variance": color_variance,
        "combined": combined,
    }


def sample_frames_evenly(frames_dir: Path, max_samples: int = 50) -> List[Tuple[int, Path]]:
    """Sample frames evenly across the video."""
    frame_files = sorted(frames_dir.glob("*.jpg"))
    total = len(frame_files)

    if total <= max_samples:
        return [(i, f) for i, f in enumerate(frame_files)]

    # Sample evenly
    step = total / max_samples
    indices = [int(i * step) for i in range(max_samples)]
    return [(idx, frame_files[idx]) for idx in indices]


def select_diverse_frames(
    scored_frames: List[Tuple[int, float, np.ndarray]],
    top_k: int = 7,
    min_distance: float = 0.3
) -> List[Tuple[int, float]]:
    """
    Select diverse frames using embeddings.
    Ensures selected frames are visually different.

    Args:
        scored_frames: List of (frame_idx, score, embedding)
        top_k: Number of frames to select
        min_distance: Minimum cosine distance between selected frames

    Returns:
        List of (frame_idx, score) tuples
    """
    if not scored_frames:
        return []

    # Sort by score descending
    sorted_frames = sorted(scored_frames, key=lambda x: x[1], reverse=True)

    selected = []
    selected_embeddings = []

    for frame_idx, score, embedding in sorted_frames:
        if len(selected) >= top_k:
            break

        # Check diversity with already selected frames
        if selected_embeddings:
            # Calculate cosine similarity with selected frames
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            similarities = []
            for sel_emb in selected_embeddings:
                sel_emb_norm = sel_emb / (np.linalg.norm(sel_emb) + 1e-8)
                similarity = np.dot(embedding_norm, sel_emb_norm)
                similarities.append(similarity)

            max_similarity = max(similarities)
            distance = 1.0 - max_similarity

            if distance < min_distance:
                continue  # Too similar to already selected frames

        selected.append((frame_idx, score))
        selected_embeddings.append(embedding)

    return selected


def check_dinov2_available() -> bool:
    """Check if DINOv2 can be loaded."""
    try:
        import torch
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            # CPU only - skip DINOv2 for performance
            return False
        # Try to load (will cache if successful)
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
        return True
    except Exception:
        return False


def load_dinov2():
    """Load DINOv2 model."""
    import torch
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


def calculate_dinov2_score(image_path: Path, model, device) -> Tuple[float, np.ndarray]:
    """
    Calculate DINOv2 objectness score and embedding.

    Returns:
        (objectness_score, cls_embedding)
    """
    import torch
    import torchvision.transforms as transforms

    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare image for DINOv2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.forward_features(image_tensor)

        # CLS token embedding (global image representation)
        cls_embedding = features[:, 0, :].cpu().numpy().flatten()

        # Patch tokens (local features)
        patch_tokens = features[:, 1:, :]

        # Objectness: variance across patches (high = diverse regions = more objects)
        objectness = patch_tokens.var(dim=1).mean().item()

    return float(objectness), cls_embedding


def suggest_optimal_frames(
    frames_dir: Path,
    top_k: int = 7,
    use_dinov2: bool = True,
    max_samples: int = 50
) -> List[Dict]:
    """
    Suggest optimal frames for annotation.

    Args:
        frames_dir: Directory containing extracted frames
        top_k: Number of frames to suggest
        use_dinov2: Whether to use DINOv2 enhancement (if available)
        max_samples: Maximum frames to analyze (for performance)

    Returns:
        List of frame info dicts with scores and indices
    """
    # Stage 1: Sample frames
    sampled = sample_frames_evenly(frames_dir, max_samples)

    if not sampled:
        return []

    # Stage 2: Basic scoring
    basic_scored = []
    for idx, path in sampled:
        scores = calculate_basic_score(path)
        basic_scored.append((idx, scores["combined"], scores))

    # Get top candidates
    basic_scored.sort(key=lambda x: x[1], reverse=True)
    candidates = basic_scored[:min(20, len(basic_scored))]

    # Stage 3: DINOv2 enhancement (optional)
    final_frames = []

    if use_dinov2 and check_dinov2_available():
        try:
            model, device = load_dinov2()

            dinov2_scored = []
            for idx, _, basic_scores in candidates:
                frame_path = frames_dir / f"{idx:05d}.jpg"
                objectness, embedding = calculate_dinov2_score(frame_path, model, device)

                # Combine basic and DINOv2 scores
                combined_score = 0.5 * basic_scores["combined"] + 0.5 * min(objectness / 10.0, 1.0)

                dinov2_scored.append((idx, combined_score, embedding))

            # Select diverse frames
            diverse_frames = select_diverse_frames(dinov2_scored, top_k)

            # Build result
            for idx, score in diverse_frames:
                basic_scores = next(s for i, _, s in candidates if i == idx)
                final_frames.append({
                    "frame_index": idx,
                    "score": score,
                    "sharpness": basic_scores["sharpness"],
                    "brightness": basic_scores["brightness"],
                    "method": "dinov2",
                })

        except Exception as e:
            print(f"DINOv2 enhancement failed: {e}, falling back to basic scoring")
            use_dinov2 = False

    # Fallback: use basic scoring only
    if not use_dinov2 or not final_frames:
        for idx, score, basic_scores in candidates[:top_k]:
            final_frames.append({
                "frame_index": idx,
                "score": score,
                "sharpness": basic_scores["sharpness"],
                "brightness": basic_scores["brightness"],
                "method": "basic",
            })

    return final_frames
