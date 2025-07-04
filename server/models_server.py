from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from server.schema_base import ResponseSchemaBase
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from server.api_router import router

class CustomException(Exception):
    http_code: int
    code: str
    message: str

    def __init__(self, http_code: int = None, code: str = None, message: str = None):
        self.http_code = http_code if http_code else 500
        self.code = code if code else str(self.http_code)
        self.message = message

async def http_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.http_code,
        content=jsonable_encoder(ResponseSchemaBase().custom_response(exc.code, exc.message))
    )

def get_application():
    app = FastAPI(
        title="Models Server",
        description="Models Server",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(CustomException, http_exception_handler)
    app.include_router(router)
    return app

app = get_application()