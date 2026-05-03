"""
RAG System — Authentication Layer
===================================
Secure user login & signup with:
  - bcrypt password hashing (never stores plain text)
  - JWT tokens for session management
  - SQLite users database (separate from rag_database.db)
  - Rate limiting to prevent brute force attacks

Install:
    pip install fastapi uvicorn bcrypt python-jose[cryptography] python-multipart

Run:
    uvicorn auth:app --reload --port 8001
"""

import sqlite3
import os
import time
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import bcrypt
from jose import JWTError, jwt

# ─────────────────────────────────────────────
# Config — change SECRET_KEY in production!
# ─────────────────────────────────────────────

SECRET_KEY      = os.environ.get("AUTH_SECRET_KEY", "change-this-secret-key-in-production-!!!!")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_HOURS = 24
USERS_DB_PATH   = "./users.db"

# Brute force protection: max 5 failed attempts per IP per 15 min
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS     = 900   # 15 minutes

app = FastAPI(title="RAG Auth API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# In-memory rate limiter { ip: [(timestamp, ...), ...] }
failed_attempts: dict = defaultdict(list)


# ─────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Encrypt with SQLite key if needed (requires pysqlcipher3 for full encryption)
    return conn


def init_users_db():
    """Create users table on first run."""
    conn = get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password_hash   TEXT    NOT NULL,
                is_active       INTEGER DEFAULT 1,
                is_admin        INTEGER DEFAULT 0,
                created_at      TEXT    NOT NULL,
                last_login      TEXT
            );

            CREATE TABLE IF NOT EXISTS login_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER REFERENCES users(id),
                ip_address  TEXT,
                success     INTEGER NOT NULL,
                attempted_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
        """)
        conn.commit()
        print(f"Users database ready: {USERS_DB_PATH}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a password with bcrypt (salt is auto-generated)."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Check plain password against stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ─────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────

def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub":      str(user_id),
        "username": username,
        "exp":      datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat":      datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency — use in protected routes."""
    return decode_token(credentials.credentials)


# ─────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────

def check_rate_limit(ip: str):
    now = time.time()
    # Remove attempts older than lockout window
    failed_attempts[ip] = [t for t in failed_attempts[ip] if now - t < LOCKOUT_SECONDS]
    if len(failed_attempts[ip]) >= MAX_FAILED_ATTEMPTS:
        wait = int(LOCKOUT_SECONDS - (now - failed_attempts[ip][0]))
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {wait} seconds."
        )


def record_failure(ip: str):
    failed_attempts[ip].append(time.time())


def clear_failures(ip: str):
    failed_attempts.pop(ip, None)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str    # can be username or email
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    user_id:      int

class UserProfile(BaseModel):
    id:         int
    username:   str
    email:      str
    is_admin:   bool
    created_at: str
    last_login: Optional[str]


# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def validate_password_strength(password: str):
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if not any(c.isdigit() for c in password):
        raise HTTPException(400, "Password must contain at least one number.")
    if not any(c.isalpha() for c in password):
        raise HTTPException(400, "Password must contain at least one letter.")


def validate_username(username: str):
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(username) > 30:
        raise HTTPException(400, "Username must be under 30 characters.")
    if not username.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "Username can only contain letters, numbers, _ and -.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_users_db()
    print("Auth API ready.")


@app.post("/auth/signup", response_model=TokenResponse)
def signup(body: SignupRequest, request: Request):
    """Register a new user. Returns a JWT token immediately."""
    validate_username(body.username)
    validate_password_strength(body.password)

    conn = get_conn()
    try:
        # Check duplicates
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? OR email=?",
            (body.username, body.email)
        ).fetchone()
        if existing:
            raise HTTPException(400, "Username or email already taken.")

        pw_hash = hash_password(body.password)
        cur = conn.execute(
            """INSERT INTO users (username, email, password_hash, created_at)
               VALUES (?, ?, ?, ?)""",
            (body.username, body.email, pw_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cur.lastrowid

        token = create_token(user_id, body.username)
        return TokenResponse(access_token=token, username=body.username, user_id=user_id)
    finally:
        conn.close()


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request):
    """Login with username/email + password. Returns JWT token."""
    ip = request.client.host
    check_rate_limit(ip)

    conn = get_conn()
    try:
        # Accept username OR email
        row = conn.execute(
            "SELECT * FROM users WHERE username=? OR email=?",
            (body.username, body.username)
        ).fetchone()

        def log_attempt(user_id, success):
            conn.execute(
                "INSERT INTO login_log (user_id, ip_address, success, attempted_at) VALUES (?,?,?,?)",
                (user_id, ip, int(success), datetime.utcnow().isoformat())
            )
            conn.commit()

        if not row or not verify_password(body.password, row["password_hash"]):
            record_failure(ip)
            log_attempt(row["id"] if row else None, False)
            raise HTTPException(401, "Invalid username or password.")

        if not row["is_active"]:
            raise HTTPException(403, "Account is deactivated.")

        clear_failures(ip)
        log_attempt(row["id"], True)

        # Update last_login
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.utcnow().isoformat(), row["id"])
        )
        conn.commit()

        token = create_token(row["id"], row["username"])
        return TokenResponse(access_token=token, username=row["username"], user_id=row["id"])
    finally:
        conn.close()


@app.get("/auth/me", response_model=UserProfile)
def me(current_user: dict = Depends(get_current_user)):
    """Get current user profile (requires valid JWT token)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, email, is_admin, created_at, last_login FROM users WHERE id=?",
            (int(current_user["sub"]),)
        ).fetchone()
        if not row:
            raise HTTPException(404, "User not found.")
        return UserProfile(**dict(row))
    finally:
        conn.close()


@app.post("/auth/logout")
def logout():
    """Frontend should discard token. Server-side: nothing to do for JWT."""
    return {"status": "logged out"}


@app.get("/auth/status")
def auth_status():
    return {"status": "ok", "service": "RAG Auth API"}


# ─────────────────────────────────────────────
# Admin — list users (admin only)
# ─────────────────────────────────────────────

@app.get("/auth/users")
def list_users(current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    try:
        row = conn.execute("SELECT is_admin FROM users WHERE id=?", (int(current_user["sub"]),)).fetchone()
        if not row or not row["is_admin"]:
            raise HTTPException(403, "Admin access required.")
        rows = conn.execute(
            "SELECT id, username, email, is_active, is_admin, created_at, last_login FROM users ORDER BY created_at DESC"
        ).fetchall()
        return {"users": [dict(r) for r in rows]}
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth:app", host="0.0.0.0", port=8001, reload=True)
