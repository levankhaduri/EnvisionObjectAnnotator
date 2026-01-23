from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class SessionCreate(BaseModel):
    name: Optional[str] = None


class Session(BaseModel):
    id: str
    name: Optional[str] = None
    status: str = "created"
    video_path: Optional[str] = None
    output_dir: Optional[str] = None
    config: Dict[str, object] = Field(default_factory=dict)


class ConfigUpdate(BaseModel):
    session_id: str
    overlap_threshold: float = 0.1
    batch_size: int = 50
    auto_fallback: bool = True
    auto_tune: bool = True
    tuning_target: float = 0.75
    tuning_reserve_gb: float = 8.0
    preview_stride: Optional[int] = None
    max_cache_frames: Optional[int] = None
    max_cache_cap: Optional[int] = None
    frame_stride: Optional[int] = None
    frame_interpolation: Optional[str] = None
    roi_enabled: bool = False
    roi_margin: float = 0.15
    roi_min_size: int = 256
    roi_max_coverage: float = 0.95
    chunk_size: Optional[int] = None
    chunk_seconds: Optional[float] = None
    chunk_overlap: int = 1
    compress_masks: Optional[bool] = None
    use_mps: bool = False
    model_key: Optional[str] = None
    export_video: bool = True
    export_elan: bool = True
    export_csv: bool = True
    multi_reference: bool = False
    reference_frames: Optional[List[int]] = None
    output_dir: Optional[str] = None


class Point(BaseModel):
    x: float
    y: float
    label: int


class AnnotationPayload(BaseModel):
    session_id: str
    frame_index: int
    object_name: str
    points: List[Point]


class ProcessingStartRequest(BaseModel):
    session_id: str


class ProcessingStatus(BaseModel):
    session_id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None


class ResultsResponse(BaseModel):
    session_id: str
    outputs: Dict[str, Optional[str]]
    profiling: Optional[Dict[str, object]] = None
    outputs_meta: Optional[Dict[str, object]] = None


class FrameExtractionRequest(BaseModel):
    session_id: str
    quality: int = 2
    auto_reference: bool = False
    auto_reference_stride: Optional[int] = None
    auto_reference_threshold: Optional[float] = None
    auto_reference_min: Optional[int] = None
    auto_reference_max: Optional[int] = None


class FrameListResponse(BaseModel):
    session_id: str
    frame_count: int
    frame_files: List[str]
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    has_thumbnails: bool = False
