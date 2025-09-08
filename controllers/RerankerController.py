from typing import List, Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from enums import ResponseStatusEnum
from services import RerankerService
from models import (
    RerankerItemModel,
    RerankerRequestModel,
    RerankerResponseModel,
    ErrorResponseModel,
)

from implementations import RerankControllerImpl


class RerankController(RerankControllerImpl):
    def __init__(self, service: RerankerService):
        self.service = service
        self.router = APIRouter()
        self.router.add_api_route("/ce/reranker", self.RerankAPI, methods=["POST"])

    async def RerankAPI(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            req = RerankerRequestModel.model_validate(payload)
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())

        query: str = req.query
        docs: List[str] = req.docs
        returnDocs: bool = req.returnDocuments
        topN: int = req.topN
        try:
            ranked_results = await self.service.Rerank(query, docs)
            items: List[Any] = []
            for index, (docIndex, doc, score) in enumerate(ranked_results):
                if returnDocs:
                    item = RerankerItemModel(
                        docIndex=docIndex, doctext=doc, score=score
                    )
                else:
                    item = RerankerItemModel(docIndex=docIndex, score=score)
                items.append(item)
                if index == topN - 1:
                    break

            resp = RerankerResponseModel(results=items, query=query)
            return JSONResponse(status_code=200, content=resp.model_dump())
        except ValueError as exc:

            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.SERVER_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=500, content=err.model_dump())
