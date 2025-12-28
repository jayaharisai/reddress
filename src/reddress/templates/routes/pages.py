"""
HTML page routes using Jinja2 templates
"""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Get absolute path to templates directory
# Go up one level from routes/ to project root, then into templates/
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Home Page"}
    )

@router.get("/dashboard-view", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    data = {"metric1": 100, "metric2": 200}
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Dashboard", "data": data}
    )
