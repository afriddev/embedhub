from pydantic import BaseModel


class RerankerRequestModel(BaseModel):
    query: str
    docs: list[str]
    returnDocuments: bool = False
    topN: int = 10


class RerankerItemModel(BaseModel):
    docIndex: int
    doctext: str | None = None
    score: float


class RerankerResponseModel(BaseModel):
    results: list[RerankerItemModel]
    query: str
