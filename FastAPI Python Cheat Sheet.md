# FastAPI Python Cheat Sheet

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"  # ASGI server
```

## Basic Setup

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Run with: uvicorn main:app --reload
```

## HTTP Methods

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items/")
async def create_item(item: dict):
    return item

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: dict):
    return {"item_id": item_id, **item}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"deleted": item_id}

@app.patch("/items/{item_id}")
async def patch_item(item_id: int, item: dict):
    return {"item_id": item_id, **item}
```

## Path Parameters

```python
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# Path parameters with types
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id: int, item_id: str):
    return {"user_id": user_id, "item_id": item_id}

# Enum path parameters
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

## Query Parameters

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Optional query parameters
from typing import Optional

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

# Multiple query parameters
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10, q: Optional[str] = None):
    items = {"skip": skip, "limit": limit}
    if q:
        items.update({"q": q})
    return items
```

## Pydantic Models (Request Body)

```python
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None
    description: Optional[str] = None

@app.post("/items/")
async def create_item(item: Item):
    return item

# Nested models
class User(BaseModel):
    name: str
    email: str

class ItemWithUser(BaseModel):
    name: str
    price: float
    owner: User

@app.post("/items-with-user/")
async def create_item_with_user(item: ItemWithUser):
    return item
```

## Response Models

```python
class ItemResponse(BaseModel):
    name: str
    price: float
    item_id: int

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: Item):
    # Process item...
    return ItemResponse(name=item.name, price=item.price, item_id=1)

# Response model with status code
from fastapi import status

@app.post("/items/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return ItemResponse(name=item.name, price=item.price, item_id=1)
```

## Error Handling

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id == "foo":
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}

# Custom exception handler
from fastapi.responses import JSONResponse

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."}
    )
```

## Dependencies

```python
from fastapi import Depends

# Simple dependency
def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Class-based dependency
class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    return commons
```

## Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Validate token here
    if token != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return {"username": "current_user"}

@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello {current_user['username']}"}
```

## File Upload

```python
from fastapi import File, UploadFile

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}

# Multiple files
@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}
```

## Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Background Tasks

```python
from fastapi import BackgroundTasks

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

## Database Integration (SQLAlchemy)

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class ItemDB(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    price = Column(Integer)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/{item_id}")
async def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

## Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.5}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"
```

## Configuration and Environment Variables

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Awesome API"
    admin_email: str
    items_per_user: int = 50

    class Config:
        env_file = ".env"

settings = Settings()

@app.get("/info")
async def info():
    return {
        "app_name": settings.app_name,
        "admin_email": settings.admin_email,
        "items_per_user": settings.items_per_user,
    }
```

## Static Files

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")
```

## Templates

```python
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/items/{id}")
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})
```

## Advanced Features

### WebSockets

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

### Tags and Metadata

```python
@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Item Foo"}]

@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]
```

### API Documentation Customization

```python
app = FastAPI(
    title="My Super Project",
    description="This is a very fancy project",
    version="2.5.0",
    docs_url="/documentation",  # Custom docs URL
    redoc_url="/redoc",         # Custom ReDoc URL
)
```

## Running the Application

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000

# With Gunicorn (production)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Common Status Codes

```python
from fastapi import status

# Use with response_model
status.HTTP_200_OK
status.HTTP_201_CREATED
status.HTTP_204_NO_CONTENT
status.HTTP_400_BAD_REQUEST
status.HTTP_401_UNAUTHORIZED
status.HTTP_403_FORBIDDEN
status.HTTP_404_NOT_FOUND
status.HTTP_422_UNPROCESSABLE_ENTITY
status.HTTP_500_INTERNAL_SERVER_ERROR
```

## Useful Imports

```python
from fastapi import (
    FastAPI, Depends, HTTPException, status, File, UploadFile,
    Request, Response, BackgroundTasks, WebSocket
)
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, BaseSettings, Field
from typing import Optional, List, Dict
```
