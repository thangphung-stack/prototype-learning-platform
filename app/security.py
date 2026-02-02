'''security is responsible for 2 jobs:
1. Password security (for lec), so when lec registers or log in:
- never store raw pwd, but a pwd hash
- later verify the login pwd against the hash
-> hash_password() + verify_password()

2. JWT for authenticated teacher requests, when login success:
- backend creates a token JWT
-lec sends it in the header Authorization: Bearer <token>
-backend verifies the token to protect endpoints line /teacher/bookings
-> create_access_token() + decode_token()

'''

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
import secrets
import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

pwd_hasher = PasswordHash.recommended()

SECRET_KEY = os.getenv("JWT_SECRET", "secret-needs-to-be-changed")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def hash_password(password: str) -> str:
    return pwd_hasher.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd_hasher.verify(password, password_hash)

def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)

    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
        "jti": secrets.token_hex(8),   
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError as e:
        raise ValueError(str(e)) from e