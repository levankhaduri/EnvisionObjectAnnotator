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
    use_mps: bool = False
    model_key: Optional[str] = None
    export_video: bool = True
    export_elan: bool = True
    export_csv: bool = True
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


class FrameExtractionRequest(BaseModel):
    session_id: str
    quality: int = 2


class FrameListResponse(BaseModel):
    session_id: str
    frame_count: int
    frame_files: List[str]
