"""
Per-request capture of pipeline step logs.

A ContextVar holds a list for the duration of one request (each request runs in
its own asyncio task, so the buffers are isolated). TraceHandler appends every
`pipeline.*` log record into the current request's buffer, so the same log lines
that print server-side are also returned to the client — no changes to pipeline
function signatures.
"""
import contextvars
import logging

_trace_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "pipeline_trace", default=None
)


def start_trace() -> list[str]:
    """Begin capturing for the current request; returns the buffer to attach later."""
    buf: list[str] = []
    _trace_var.set(buf)
    return buf


class TraceHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        buf = _trace_var.get()
        if buf is None or not record.name.startswith("pipeline."):
            return
        stage = record.name.split(".")[-1]
        buf.append(f"[{stage}] {record.getMessage()}")
