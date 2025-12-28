"""
HTML page routes using Jinja2 templates
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Initialize templates (relative to project root)
templates = Jinja2Templates(directory="templates")

@router.get("/")
async def home(request: Request):
    return {
        "success": True
    }