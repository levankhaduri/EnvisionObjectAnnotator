import threading
from typing import Dict
from .schemas import Session, ProcessingStatus


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.sessions: Dict[str, Session] = {}
        self.processing: Dict[str, ProcessingStatus] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.cancel_flags: Dict[str, bool] = {}

    def create_session(self, session: Session) -> None:
        with self._lock:
            self.sessions[session.id] = session
            self.processing[session.id] = ProcessingStatus(
                session_id=session.id,
                status="idle",
                progress=0.0,
            )

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            return self.sessions[session_id]

    def update_session(self, session_id: str, **kwargs) -> Session:
        with self._lock:
            session = self.sessions[session_id]
            updated = session.model_copy(update=kwargs)
            self.sessions[session_id] = updated
            return updated

    def set_processing(self, status: ProcessingStatus) -> None:
        with self._lock:
            self.processing[status.session_id] = status

    def get_processing(self, session_id: str) -> ProcessingStatus:
        with self._lock:
            return self.processing[session_id]

    def set_thread(self, session_id: str, thread: threading.Thread) -> None:
        with self._lock:
            self.threads[session_id] = thread

    def get_thread(self, session_id: str):
        with self._lock:
            return self.threads.get(session_id)

    def clear_thread(self, session_id: str) -> None:
        with self._lock:
            if session_id in self.threads:
                del self.threads[session_id]

    def request_cancel(self, session_id: str) -> None:
        with self._lock:
            self.cancel_flags[session_id] = True

    def is_cancelled(self, session_id: str) -> bool:
        with self._lock:
            return self.cancel_flags.get(session_id, False)

    def clear_cancel(self, session_id: str) -> None:
        with self._lock:
            if session_id in self.cancel_flags:
                del self.cancel_flags[session_id]


state = AppState()
