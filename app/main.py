from fastapi import Depends, FastAPI

from .dependencies import get_query_token, get_token_header
from .internal import admin
from .routers import nlp, users

app = FastAPI()

app.include_router(users.router)
app.include_router(nlp.router)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],

    responses={418: {"description": "I'm a teapot"}},
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
