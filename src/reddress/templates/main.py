import uvicorn
from .app import app

def server():
    uvicorn.run(
        "reddress.templates.app:app",
        host="0.0.0.0",
        port=80,
        reload=True  
    )
