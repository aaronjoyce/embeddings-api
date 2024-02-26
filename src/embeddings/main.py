from fastapi import HTTPException, status

from fastapi.responses import JSONResponse

from fastapi import Request

from embeddings.app.factory import create_app
from embeddings.app.config import settings

from fastapi.security.utils import get_authorization_scheme_param

app = create_app()


@app.middleware("http")
async def authentication_middleware(request: Request, call_next):
    if settings.ADMIN_SECRET_KEY is not None:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": [{"msg": "Missing basic authorization token"}]}
            )

        if scheme.lower() != "basic":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": [{
                    "msg": "Authorization header value does not match the expected 'Basic' auth scheme"
                }]}
            )

        api_key = authorization.split(' ')[-1]
        if api_key != settings.ADMIN_SECRET_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": [{"msg": "Invalid authorization token provided"}]}
            )

    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=True,
        port=int("8000")
    )
