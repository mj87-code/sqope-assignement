import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from api.trace import TraceHandler

# Surface pipeline step logs on stdout (visible via `docker-compose logs -f api`)
# and capture them per-request so they can be returned in the response.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logging.getLogger().addHandler(TraceHandler())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify DB connectivity at startup — fail fast rather than serving broken requests
    from sqlalchemy import text

    from database.connection import dispose_engines, get_async_session
    try:
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(f"Database unreachable at startup: {exc}") from exc
    yield
    await dispose_engines()


app = FastAPI(
    title="Sqope Due Diligence QA",
    description="PDF-based question answering for financial due diligence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log full detail server-side; the client only gets a generic message —
    # the raw exception can carry internal details (DB/driver errors, internal
    # paths) that shouldn't reach an external caller of this due-diligence API.
    logging.getLogger("api").exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. Please try again or contact support."},
    )
