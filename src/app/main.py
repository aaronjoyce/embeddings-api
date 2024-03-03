from fastapi import status, Request

from fastapi.responses import JSONResponse

from app.factory import create_app
from app.config import settings
from app.exceptions import (
    NotFoundException,
    UnknownThirdPartyException,
    EmbeddingDimensionalityException,
    EnvironmentVariableConfigException
)

from fastapi.security.utils import get_authorization_scheme_param

app = create_app()


@app.exception_handler(UnknownThirdPartyException)
async def unknown_third_party_exception(request: Request, exc: UnknownThirdPartyException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": str(exc)
        }
    )


@app.exception_handler(EmbeddingDimensionalityException)
async def embedding_dimensionality_exception(request: Request, exc: EmbeddingDimensionalityException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc)
        }
    )


@app.exception_handler(NotFoundException)
async def not_found_exception_handler(request: Request, exc: NotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": str(exc)
        }
    )


@app.exception_handler(EnvironmentVariableConfigException)
async def environment_variable_config_exception_handle(request: Request, exc: EnvironmentVariableConfigException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": str(exc)
        }
    )


@app.middleware("http")
async def authentication_middleware(request: Request, call_next):
    if settings.ADMIN_SECRET_KEY is not None:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing basic authorization token"}
            )

        if scheme.lower() != "basic":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header value does not match the expected 'Basic' auth scheme"}
            )

        api_key = authorization.split(' ')[-1]
        if api_key != settings.ADMIN_SECRET_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid authorization token provided"}
            )

    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        reload=True,
        port=settings.API_PORT
    )
