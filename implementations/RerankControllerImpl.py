from abc import ABC, abstractmethod
from fastapi import Request
from fastapi.responses import JSONResponse


class RerankControllerImpl(ABC):

    @abstractmethod
    async def RerankAPI(self, request: Request) -> JSONResponse:
        pass
