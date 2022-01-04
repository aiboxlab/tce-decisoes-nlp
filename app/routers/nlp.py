from typing import List

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from fastapi.encoders import jsonable_encoder
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


class Features(BaseModel):
    text: str
    dictionary: dict
    html_summary: List
    summary: List
    html: str
    sentiments: dict
    topics: dict


# {
#             "text": self.get_text(),
#             "dictionary": self.get_dictionary(),
#             "html_summary": html_summary,
#             "summary": self.summary,
#             "html": self.get_glossary_html(),
#             "sentiments": self.get_sentiments(self.summary),
#             "topics": self.get_topics(self.summary)
#         }


@router.get("/")
async def read_items():
    return fake_items_db


@router.post("/summarize", response_model=List[str])
async def summarize(request: RequestBody = Body(...)):
    doc = Document(request.text)
    return doc.get_summary(3)


@router.post("/features")
async def summarize(request: RequestBody = Body(...)):
    doc = Document(request.text)
    return jsonable_encoder(doc.get_features_dict())


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
