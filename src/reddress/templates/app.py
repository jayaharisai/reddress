from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routes import dashboard, pages

# Get base directory (where app.py is located)
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Multi-Module API",
    description="FastAPI server with multiple route modules",
    version="1.0.0"
)

# Initialize templates and static files with absolute paths
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include routers
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(pages.router, tags=["Pages"])
