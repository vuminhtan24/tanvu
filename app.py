from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from routes.chicken import router as chicken_router
from routes.grade    import router as grade_router

app = FastAPI(title="Vũ Minh Tân — AI Portfolio", version="1.0.0")

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Include API routers ───────────────────────────────────────────────────────
app.include_router(chicken_router)
app.include_router(grade_router)

# ── Helper để đọc template ────────────────────────────────────────────────────
def read_template(name: str) -> str:
    path = os.path.join("templates", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── Page routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return read_template("index.html")

@app.get("/chicken", response_class=HTMLResponse)
async def chicken_page():
    return read_template("chicken.html")

@app.get("/grade", response_class=HTMLResponse)
async def grade_page():
    return read_template("grade.html")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)