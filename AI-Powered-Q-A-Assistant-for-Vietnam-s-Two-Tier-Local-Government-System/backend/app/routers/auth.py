"""
Authentication router — register and login endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from app.auth import hash_password, verify_password, create_access_token, get_current_user
from app.models import User
from app import crud

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Check if email already exists
    existing = crud.get_user_by_email(db, req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create user
    hashed = hash_password(req.password)
    user = crud.create_user(db, email=req.email, password_hash=hashed, full_name=req.full_name)

    # Generate token
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT token."""
    user = crud.get_user_by_email(db, req.email)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user profile."""
    return current_user
