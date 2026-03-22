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
    output_dir: Optional[str] = None
    process_start_frame: Optional[int] = None
    process_end_frame: Optional[int] = None
    enable_bidirectional: bool = True
    enhance_target: bool = False


class Point(BaseModel):
    x: float
    y: float
    label: int


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationPayload(BaseModel):
    session_id: str
    frame_index: int
    object_name: str
    previous_object_name: Optional[str] = None
    points: List[Point]
    bbox: Optional[BBox] = None


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
    file_exists: Optional[Dict[str, bool]] = None
    profiling: Optional[Dict[str, object]] = None
    outputs_meta: Optional[Dict[str, object]] = None


class FrameExtractionRequest(BaseModel):
    session_id: str
    quality: int = 2
    start_time: Optional[float] = None  # trim start in seconds
    end_time: Optional[float] = None    # trim end in seconds


class FrameListResponse(BaseModel):
    session_id: str
    frame_count: int
    frame_files: List[str]
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    has_thumbnails: bool = False


class SampleClipRequest(BaseModel):
    session_id: str
    duration_seconds: float = 10.0


class DetectGreyRequest(BaseModel):
    session_id: str
    max_frames: int = 300  # scan up to this many frames
    variance_threshold: float = 100.0  # below this = grey/uniform


class DetectGreyResponse(BaseModel):
    first_valid_frame: int
    first_valid_time: float  # in seconds
    frames_scanned: int


class SuggestedFrame(BaseModel):
    frame_index: int
    score: float
    sharpness: float
    brightness: float
    method: str  # "basic" or "dinov2"


class FrameSuggestionResponse(BaseModel):
    session_id: str
    suggested_frames: List[SuggestedFrame]
    total_analyzed: int
    method_used: str
