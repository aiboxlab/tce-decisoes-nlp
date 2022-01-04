from typing import List

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from pydantic import BaseModel

from ..tce_nlp import Document
from ..dependencies import get_token_header

router = APIRouter(
    prefix="/tce_nlp",
    tags=["TCE NLP"],

    responses={404: {"description": "Not found"}},
)

fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


class RequestBody(BaseModel):
    text: str


@router.get("/")
async def read_items():
    return fake_items_db


@router.post("/summarize", response_model=List[str])
async def summarize(request: RequestBody = Body(...)):

    doc = Document(request.text)
    return doc.get_summary(3)


@router.put(
    "/{item_id}",
    tags=["custom"],
    responses={403: {"description": "Operation forbidden"}},
)
async def update_item(item_id: str):
    if item_id != "plumbus":
        raise HTTPException(
            status_code=403, detail="You can only update the item: plumbus"
        )
    return {"item_id": item_id, "name": "The great Plumbus"}
