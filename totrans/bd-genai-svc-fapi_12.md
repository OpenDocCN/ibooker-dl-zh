# Chapter 8\. Authentication and Authorization

So far, you’ve built GenAI services that can interact with databases, stream model responses, and handle concurrent users.

Your services are now up and running, but since they’re not protected from attackers or malicious users, deploying them to production may prove problematic.

In this chapter, you’ll learn how to secure your services with an authentication layer and implement authorization guards to protect sensitive resources from nonprivileged users.

To achieve this, we’re going to explore various authentication and authorization patterns then implement JWT and identity-based authentication with role-based access control.

# Authentication and Authorization

Before talking about authentication methods, let’s briefly clarify that authentication and authorization are two separate concepts that are often interchangeably used by mistake.

According to the OWASP definition:^([1](ch08.html#id970))

> *Authentication* is the process of verifying that an individual, entity, or website is who or what it claims to be by determining the validity of one or more authenticators (like passwords, fingerprints, or security tokens) that are used to back up this claim.

On the other hand, the National Institute of Standards and Technology (NIST) defines authorization as:

> A process for verifying that a requested action or service is approved for a specific entity.

While authentication is about verifying the identity, authorization focuses on verifying permissions of an identity to access or mutate resources.

###### Tip

An analogy that might clarify this distinction is passing through passport control at an airport. Authentication is like presenting your passport at immigration, while authorization is like having the right visa to enter a country, specifying the duration of your stay and permitted activities once you enter.

Let’s discuss authentication methods in more detail before diving into authorization later in the chapter.

# Authentication Methods

There are several authentication mechanisms that you can implement in your GenAI services to secure them by identity verification.

Depending on your security requirements, application environment, budget, and project timelines, you may decide to adopt one or more of the following authentication mechanisms:

Basic

Requiring the use of credentials such as username and password to verify identity.

JSON Web Tokens (JWT)

Requiring the use of *access tokens* to verify identity. You can think of access tokens like cinema tickets that dictate whether you can access the screens and which screen you’re visiting and where you’re sitting.

OAuth

Verifying an identity via an identity provider using the *OAuth2* standard.

Key-based

Using a private and public key pair to authenticate an identity. Instead of tokens, the authorization server issues a public key to the client and stores a copy of a linked private key that it can use later for verification.^([2](ch08.html#id971))

[Figure 8-1](#authentication_methods) shows the data flow of the aforementioned authentication methods in more detail.

![bgai 0801](assets/bgai_0801.png)

###### Figure 8-1\. Authentication methods

Being aware of authentication mechanisms, it can still be challenging to decide on the method to adopt when addressing your security requirements. To assist with the selection task, [Table 8-1](#authentication_methods_comparison) compares the aforementioned authentication methods.

Table 8-1\. Comparison of authentication methods

| Type | Benefits | Limitations | Use cases |
| --- | --- | --- | --- |
| Basic |  
*   Simplicity

*   Fast to implement

*   Easy to understand

 | Sends credentials in plain text |  
*   Prototypes

*   Internal or nonproduction environments

 |
| Token |  
*   Scalability

*   Decoupling facilitates implementation of microservice architectures

*   Tokens can be signed and encrypted for higher security

*   Highly customizable

*   Self-contained reducing database round-trips

*   Can be passed in HTTP headers

 |  
*   Constant need to regenerate short-lived tokens

*   Complexity of client-side token storage

*   Tokens can get large, consuming excess bandwidth

*   Stateless tokens can make multi-step applications hard to implement

*   Client-side misconfigurations can compromise tokens

 |  
*   Single-page and mobile applications

*   Applications requiring custom authentication flows

*   REST APIs

 |
| OAuth |  
*   Delegates authentication to external providers

*   Based on a standard (OAuth2) and battle-tested for enterprise scenarios

*   Access to external resources on behalf of the user

 |  
*   Complex to understand and implement

*   Each identity provider may implement the OAuth flow slightly differently

 |  
*   Applications requiring user data from external identity providers such as GitHub, Google, or Microsoft

*   Enterprise applications that require SSO with their own identity provider(s)

 |
| Key-based |  
*   Similar authentication mechanism to Secure Shell (SSH) access

 |  
*   Managing and keeping private keys secure can be complex

*   Compromised keys can create security risks

*   Scalability issues

 |  
*   Small applications

*   Applications within internal environments

 |

You should now feel confident in deciding the appropriate authentication mechanism to adopt. In the next section, you’re going to implement basic, JWT, and OAuth authentication for your GenAI to fully understand the underlying components and their interactions.

## Basic Authentication

In basic authentication, the client provides a username and password when making a request to access resources from the server. It is the simplest technique as it won’t require cookies, session identifiers, or any login forms to be implemented. Because of its simplicity, basic authentication is ideal for sandbox environments and when prototyping. However, avoid using it in production environments as it transmits usernames and passwords in plain text on every request, making it highly vulnerable to interception attacks.

To perform an authenticated request via basic authentication, you must add an `Authorization` header with a value of `Basic <credentials>` for the server to successfully authenticate it. The `<credentials>` value must be a *Base64* encoding of the username and password joined by a single colon (i.e., `base64.encode(ali:secretpassword)`.

In FastAPI, you can protect an endpoint with basic authentication, as shown in [Example 8-1](#basic_authentication_endpoint).

##### Example 8-1\. Implementing basic authentication in FastAPI

```py
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI()
security = HTTPBasic() ![1](assets/1.png)
username_bytes = b"ali"
password_bytes = b"secretpassword"

def authenticate_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str:
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("UTF-8"), username_bytes ![2](assets/2.png)
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("UTF-8"), password_bytes ![2](assets/2.png)
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException( ![3](assets/3.png)
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

AuthenticatedUserDep = Annotated[str, Depends(authenticate_user)] ![4](assets/4.png)

@app.get("/users/me")
def get_current_user_controller(username: AuthenticatedUserDep): ![4](assets/4.png)
    return {"message": f"Current user is {username}"}
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO1-1)

FastAPI has implemented several HTTP security mechanisms including `HTTP​Ba⁠sic` that can leverage the FastAPI’s dependency injection system.

[![2](assets/2.png)](#co_authentication_and_authorization_CO1-2)

Use the `secrets` built-in library to compare the provided username and password with the server’s values. Using `secrets.compare_digest()` ensures the duration of checking operations remain consistent no matter what the inputs are to avoid *timing attacks*.^([3](ch08.html#id972))

Note that `secrets.compare_digest()` can only accept byte or string inputs containing ASCII characters (i.e., English-only characters). To handle other characters, you will need to encode the inputs with `UTF-8` to bytes first before performing the credential checks.

[![3](assets/3.png)](#co_authentication_and_authorization_CO1-4)

Return a standardized authorization `HTTPException` compliant with security standards that browsers understand so that they show the login prompt again to the user. The exception message must be generic to avoid leaking any sensitive information, such as the existence of a user account, to attackers.

[![4](assets/4.png)](#co_authentication_and_authorization_CO1-5)

Using the `HTTPBasic` with `Depends()` returns the `HTTPBasicCredentials` object that contains the provided username.

Injecting a security dependency to any FastAPI endpoint will protect it with the implemented authentication. You can experience this yourself now by visiting the `/docs` page and sending a request to the `/users/me` endpoint.

The endpoint will show a *lock* icon in front of it, and you should see a sign-in alert when making a request, asking you to provide credentials, as you can see in [Figure 8-2](#basic_authentication).

![bgai 0802](assets/bgai_0802.png)

###### Figure 8-2\. Basic authentication in FastAPI

Well done! In 25 lines of code, you managed to implement a basic form of authentication to protect an endpoint. You can now use basic authentication in your own prototypes and development servers.

Bear in mind, you should avoid adopting the basic authentication mechanism in production-grade GenAI services. A better and more secure alternative for public-facing services is *JWT authentication*. It eliminates the need for server-side sessions by storing all authentication details within a token. It also maintains data integrity and works across different domains with a widely accepted standard.

## JSON Web Tokens (JWT) Authentication

Now that you’re more familiar with basic concepts of authentication, let’s implement a more complex but secure JWT authentication layer to your FastAPI service. As part of this, you’ll also refactor your existing endpoints to combine them under a separate resource API router to group, name, tag, and protect multiple endpoints at once.

### What is JWT?

JWTs are a URL-safe and compact way of asserting claims between applications via tokens.

These tokens consist of three parts:

Headers

Specify the token type and signing algorithm in addition to the datetime and the issuing authority.

Payload

Specify the body of the token representing the claims on the resource alongside additional metadata.

Signature

The function that creates the token will also sign it using the *encoded payload*, *encoded headers*, a *secret*, and the signing algorithm.

###### Tip

The `base64` encoding algorithm is often used to encode and decode data for compactness and URL safety.

[Figure 8-3](#jwt) shows what a typical JWT looks like.

![bgai 0803](assets/bgai_0803.png)

###### Figure 8-3\. JWT components (Source: [jwt.io](https://jwt.io))

JWTs are secure, compact, and convenient since they can hold all the information needed to perform user authentication, avoiding the need for multiple database round-trips. In addition, due to their compactness, you can transfer them across the network using the HTTP `POST` body, headers, or URL parameters.

### Getting started with JWT authentication

To get started with implementing the JWT authentication mechanism in FastAPI, you need to install the `passlib` and `python-jose` dependencies:

```py
$ pip install passlib python-jose
```

With the dependencies installed, you will then need tables in the database to store the generated users and associated token data. For data persistence, let’s migrate the database to create the `users` and `tokens` tables, as shown in [Figure 8-4](#erd).

![bgai 0804](assets/bgai_0804.png)

###### Figure 8-4\. Entity relationship diagram of `users` and `tokens` tables

If you look at [Figure 8-4](#erd), you will spot that the `tokens` table has a one-to-many relationship with the `users` table. You can use the token records to track successful login attempts for each user and to revoke access if needed.

Next, let’s define the required SQLAlchemy models and Pydantic schemas for database queries and data validation, as shown in Examples [8-2](#user_models) and [8-3](#users_schema).

##### Example 8-2\. Declare user SQLAlchemy ORM models

```py
# entities.py

import uuid
from datetime import UTC, datetime
from sqlalchemy import Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(length=255), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(length=255))
    is_active: Mapped[bool] = mapped_column(default=True)
    role: Mapped[str] = mapped_column(default="USER")
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    __table_args__ = (Index("ix_users_email", "email"),)
```

You will be using the ORM models at the data access layer while the Pydantic schemas will validate incoming and outgoing authentication data at the endpoint layer.

##### Example 8-3\. Declare user Pydantic schemas with username and password field validators

```py
# schemas.py

from datetime import datetime
from typing import Annotated
from pydantic import (UUID4, AfterValidator, BaseModel, ConfigDict, Field,
                      validate_call)

@validate_call
def validate_username(value: str) -> str: ![1](assets/1.png)
    if not value.isalnum():
        raise ValueError("Username must be alphanumeric")
    return value

@validate_call
def validate_password(value: str) -> str: ![1](assets/1.png)
    validations = [
        (
            lambda v: any(char.isdigit() for char in v),
            "Password must contain at least one digit",
        ),
        (
            lambda v: any(char.isupper() for char in v),
            "Password must contain at least one uppercase letter",
        ),
        (
            lambda v: any(char.islower() for char in v),
            "Password must contain at least one lowercase letter",
        ),
    ]
    for condition, error_message in validations:
        if not condition(value):
            raise ValueError(error_message)
    return value

ValidUsername = Annotated[
    str, Field(min_length=3, max_length=20), AfterValidator(validate_username)
]
ValidPassword = Annotated[
    str, Field(min_length=8, max_length=64), AfterValidator(validate_password)
]

class UserBase(BaseModel):
    model_config = ConfigDict(from_attributes=True) ![2](assets/2.png)

    username: ValidUsername
    is_active: bool = True
    role: str = "USER"

class UserCreate(UserBase): ![3](assets/3.png)
    password: ValidPassword

class UserInDB(UserBase): ![4](assets/4.png)
    hashed_password: str

class UserOut(UserBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO2-1)

Validate both username and password to enforce higher security requirements.

[![2](assets/2.png)](#co_authentication_and_authorization_CO2-3)

Allow Pydantic to read SQLAlchemy ORM model attributes instead of having to manually populate Pydantic schemas from SQLAlchemy models.

[![3](assets/3.png)](#co_authentication_and_authorization_CO2-4)

Use inheritance to declare several Pydantic schemas based on a user base model.

[![4](assets/4.png)](#co_authentication_and_authorization_CO2-5)

Create a separate schema that accepts the `hashed_password` field to be used only for creating new user records during the registration process. All other schemas must skip storing this field to eliminate the risk of password leakage.

Creating the token models and schemas is fairly similar, as you can see in [Example 8-4](#token_models).

##### Example 8-4\. Declare token ORM models and Pydantic schemas

```py
# entities.py

from datetime import UTC, datetime
from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Token(Base):
    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    ip_address: Mapped[str | None] = mapped_column(String(length=255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    user = relationship("User", back_populates="tokens")

    __table_args__ = (
        Index("ix_tokens_user_id", "user_id"),
        Index("ix_tokens_ip_address", "ip_address"),
    )

class User(Base):
    __tablename__ = "users"
    # other columns...

    tokens = relationship(
        "Token", back_populates="user", cascade="all, delete-orphan"
    )

# schemas.py

from datetime import datetime
from pydantic import BaseModel

class TokenBase(BaseModel):
    user_id: int
    expires_at: datetime
    is_active: bool = True
    ip_address: str | None = None

class TokenCreate(TokenBase):
    pass

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "Bearer"
```

Next, let’s auto-generate a migration file using the `alembic revision --autogenerate -m "create users and tokens tables` command so that you can specify the details of both tables by following [Example 8-5](#users_migration).

##### Example 8-5\. Database migration to create the `users` and `tokens` tables

```py
"""create users and tokens tables

Revision ID: 1234567890ab
Revises:
Create Date: 2025-01-28 12:34:56.789012

"""

from datetime import UTC, datetime
import sqlalchemy as sa
from alembic import op

...

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(as_uuid=True)), ![1](assets/1.png)
        sa.Column("email", sa.String(length=255)),
        sa.Column("hashed_password", sa.String(length=255)), ![2](assets/2.png)
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.sql.expression.true()
        ), ![3](assets/3.png)
        sa.Column("role", sa.String(), server_default=sa.text("USER")), ![4](assets/4.png)
        sa.Column("created_at", sa.DateTime(), default=datetime.now(UTC)),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC), ![5](assets/5.png)
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.Index("ix_users_email", "email"), ![6](assets/6.png)
    )

    op.create_table(
        "tokens",
        sa.Column("id", sa.UUID(as_uuid=True)), ![1](assets/1.png)
        sa.Column("user_id", sa.Integer()),
        sa.Column("expires_at", sa.DateTime()), ![7](assets/7.png)
        sa.Column("is_active", sa.Boolean(), default=True), ![8](assets/8.png)
        sa.Column("ip_address", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=datetime.now(UTC)),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC), ![9](assets/9.png)
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_tokens_user_id", "user_id"),
        sa.Index("ix_tokens_ip_address", "ip_address"), ![10](assets/10.png)
    )

def downgrade():
    op.drop_table("tokens")
    op.drop_table("users")
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO3-1)

Automatically generate universally unique identifiers (UUIDs) in the database layer for user and token records to prevent attackers from guessing identifiers of sensitive resources (i.e., user or token records).

[![2](assets/2.png)](#co_authentication_and_authorization_CO3-2)

Avoid storing raw password strings in the database to reduce security vulnerabilities.

[![3](assets/3.png)](#co_authentication_and_authorization_CO3-3)

Add the ability to enable or disable account access.

[![4](assets/4.png)](#co_authentication_and_authorization_CO3-4)

Add the ability to specify user roles such as `USER` and `ADMIN` for managing access levels of an account. Authorization checks will use the `role` field to manage access of privileged resources.

[![5](assets/5.png)](#co_authentication_and_authorization_CO3-5)

Auto-timestamp user creation and updates for monitoring and security purposes.

[![6](assets/6.png)](#co_authentication_and_authorization_CO3-6)

Add a unique constraint and a secondary index on the email field to optimize user queries by email and eliminate the possibility of creating duplicate email accounts.

[![7](assets/7.png)](#co_authentication_and_authorization_CO3-8)

Tokens must expire after a short period of time to reduce the time window that exposed tokens may be misused by attackers.

[![8](assets/8.png)](#co_authentication_and_authorization_CO3-9)

Add the ability to disable tokens that should no longer be valid for either being exposed or if a user has logged out.

[![9](assets/9.png)](#co_authentication_and_authorization_CO3-10)

Track the token creation and update times for monitoring and security.

[![10](assets/10.png)](#co_authentication_and_authorization_CO3-11)

Create secondary indexes on `user_id` and `ip_address` fields to optimize token queries by these fields.

Now, run the `alembic upgrade head` command to execute the migration in [Example 8-5](#users_migration) against your database and create both `users` and `tokens` tables.

With the ORM models and Pydantic schemas declared, you can focus on the core authentication mechanism logic.

[Figure 8-5](#jwt_architecture) shows the architecture of the JWT authentication system you’re going to implement in your FastAPI GenAI service.

![bgai 0805](assets/bgai_0805.png)

###### Figure 8-5\. JWT authentication system architecture

In the following code examples, you will see how to implement the core authentication flows starting with user registration and JWT generation.

### Hashing and salting

The first step after creating the `users` and `tokens` tables in the database is to store new users in the database upon registration. However, you should avoid storing passwords in plain form, because if the database is compromised, the attackers will have every user’s credential.

Instead, the authentication mechanism will leverage a *hashing algorithm* that converts plain passwords into an encoded string that can’t be decoded back into its original form. Since the decoding process isn’t reversible, cryptographic hashing algorithms differ from standard encoding/decoding functions such as Base64.

While storing hashed passwords is more secure than storing plain passwords, it doesn’t provide enough protection. If such a database of hashed passwords falls into the hands of attackers, they can use a precomputed hash tables—​commonly referred to as *rainbow tables*. Attackers can use rainbow tables to brute-force their way into your system by recovering plaintext passwords. To protect against these brute-force attacks, you also need to introduce an element of randomness to your hashing process using a technique termed *salting*.

With salts, the cryptographic hashing algorithm produces different hashed passwords, even though the users may register with common, compromised, or duplicate passwords.

###### Warning

Password hashing with a random salt protects against brute-force attacks using rainbow tables. However, it doesn’t protect against *password spraying*, where attackers use a database of common passwords, or *credential stuffing*, where attackers enumerate on a list of compromised passwords.

During salting, the hashing function generates a random salt that appends to the plain password prior to hashing and then generates a hashed password.^([4](ch08.html#id982)) Before storing the hashed password in the database, the salt is prefixed to the hashed password for later retrieval during verification.

When registered users try to log in, they have to supply the same password they used to create their account. During the password verification process, the password that the user provides is hashed using the same salt that was used during registration which is retrieved from the database. If the generated hashed password is exactly identical to the hashed password in the database, then the user is authenticated. Otherwise, you can safely assume that wrong credentials have been supplied.

The salting and hashing are powerful techniques that prevent attackers from brute-forcing their way into your system with rainbow tables. You can see the full hashing and salting process in [Figure 8-6](#password_hashing).

![bgai 0806](assets/bgai_0806.png)

###### Figure 8-6\. Password hash salting mechanism

The password service shown in [Figure 8-6](#password_hashing) is implemented as `PasswordService` in [Example 8-6](#password_service).

##### Example 8-6\. Implement a password service

```py
# services/auth.py

from fastapi.security import HTTPBearer
from passlib.context import CryptContext

class PasswordService:
    security = HTTPBearer()
    pwd_context = CryptContext(schemes=["bcrypt"]) ![1](assets/1.png)

    async def verify_password(
        self, password: str, hashed_password: str
    ) -> bool:
        return self.pwd_context.verify(password, hashed_password) ![2](assets/2.png)

    async def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password) ![2](assets/2.png)
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO4-1)

Create an `AuthService` with a secret and password context managed by the `bcrypt` library that will handle all the user password hashing and verification.

[![2](assets/2.png)](#co_authentication_and_authorization_CO4-2)

Use `bcrypt`’s cryptography algorithm and application secret to hash and verify passwords.

The `bcrypt` cryptographic library provides the core functionality of the `Password​Ser⁠vice` for hashing and verifying passwords. Using this service, requests can now be authenticated.

If a request can’t be authenticated, you will also need to raise authorization-related exceptions, as shown in [Example 8-7](#auth_exceptions).

##### Example 8-7\. Create authentication exceptions

```py
# exceptions.py

from fastapi import HTTPException, status

UnauthorizedException = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)

AlreadyRegisteredException = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Username already registered",
)
```

The two most common authorization HTTP exceptions you will raise are related to unauthorized access or bad requests due to using already used usernames.

Once you have checked a user’s identity via their credentials, you will need to issue them an *access token*. These tokens should be short-lived to reduce the time-window that an attacker can use the token to access resources if the token is stolen.

To reduce the size footprints of the tokens and protect against *token forgery*, the token service will sign (using a secret) and encode the token payloads with an encoding such as Base64. The payload will normally contain the user’s details such as their ID, role, issuance system, and expiry dates.

The token service can also decode the payload of received tokens and check their validity during the authentication process.

Finally, the token service will also require database access to store and retrieve tokens to perform its functions. Therefore, it should inherit a `TokenRepository`, as shown in [Example 8-8](#token_repository).

##### Example 8-8\. Implementing token repository

```py
# repositories.py

from entities import Token
from repositories.interfaces import Repository
from schemas import TokenCreate, TokenUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class TokenRepository(Repository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take: int) -> list[Token]:
        async with self.session.begin():
            result = await self.session.execute(
                select(Token).offset(skip).limit(take)
            )
        return [r for r in result.scalars().all()]

    async def get(self, token_id: int) -> Token | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(Token).where(Token.id == token_id)
            )
        return result.scalars().first()

    async def create(self, token: TokenCreate) -> Token:
        new_token = Token(**token.dict())
        async with self.session.begin():
            self.session.add(new_token)
            await self.session.commit()
            await self.session.refresh(new_token)
        return new_token

    async def update(
        self, token_id: int, updated_token: TokenUpdate
    ) -> Token | None:
        token = await self.get(token_id)
        if not token:
            return None
        for key, value in updated_token.dict(exclude_unset=True).items():
            setattr(token, key, value)
        async with self.session.begin():
            await self.session.commit()
            await self.session.refresh(token)
        return token

    async def delete(self, token_id: int) -> None:
        token = await self.get(token_id)
        if not token:
            return
        async with self.session.begin():
            await self.session.delete(token)
            await self.session.commit()
```

With the `TokenRepository` implemented, you can now develop the `TokenService`, as shown in [Example 8-9](#token_service).

##### Example 8-9\. Implement a token service by inheriting the token repository

```py
# services/auth.py

from datetime import UTC, datetime, timedelta
from exceptions import UnauthorizedException
from jose import JWTError, jwt
from pydantic import UUID4
from repositories import TokenRepository
from schemas import TokenCreate, TokenUpdate

class TokenService(TokenRepository):
    secret_key = "your_secret_key"
    algorithm = "HS256"
    expires_in_minutes = 60 ![1](assets/1.png)

async def create_access_token(
    self, data: dict, expires_delta: timedelta | None = None ![2](assets/2.png)
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=self.expires_in_minutes)
    token_id = await self.create(TokenCreate(expires_at=expire)) ![3](assets/3.png)
    to_encode.update(
        {"exp": expire, "iss": "your_service_name", "sub": token_id} ![4](assets/4.png)
    )
    encoded_jwt = jwt.encode(
        to_encode, self.secret_key, algorithm=self.algorithm ![5](assets/5.png)
    )
    return encoded_jwt

async def deactivate(self, token_id: UUID4) -> None:
    await self.update(TokenUpdate(id=token_id, is_active=False))

def decode(self, encoded_token: str) -> dict:
    try:
        return jwt.decode(
            encoded_token, self.secret_key, algorithms=[self.algorithm]
        )
    except JWTError:
        raise UnauthorizedException

async def validate(self, token_id: UUID4) -> bool:
    return (token := await self.get(token_id)) is not None and token.is_active
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO5-1)

Implement a `TokenService` for issuing and checking authentication tokens. Configurations are shared across all instances of the service.

[![2](assets/2.png)](#co_authentication_and_authorization_CO5-2)

Generate access tokens based on data provided to the token service with expiry dates.

[![3](assets/3.png)](#co_authentication_and_authorization_CO5-3)

Create a token record in the database and get a unique identifier.

[![4](assets/4.png)](#co_authentication_and_authorization_CO5-4)

The access token must expire within an hour, so the `exp` calculated field will be used to check token validity.

[![5](assets/5.png)](#co_authentication_and_authorization_CO5-5)

Encode the generated token into an encoded string using the `base64` algorithm.

Now that you have a `PasswordService` and a `TokenService`, you can complete the core JWT authentication mechanism with a dedicated higher-level `AuthService`.

[Example 8-10](#auth_service) shows the implementation of the `AuthService` class that contains several dependency functions for registering users, issuing access tokens, and protecting your API routes.

##### Example 8-10\. Implement an auth service to handle higher-level authentication logic

```py
# services/auth.py

from typing import Annotated
from databases import DBSessionDep
from entities import Token, User, UserCreate, UserInDB
from exceptions import AlreadyRegisteredException, UnauthorizedException
from fastapi import Depends
from fastapi.security import (HTTPAuthorizationCredentials, HTTPBearer,
                              OAuth2PasswordRequestForm)
from services.auth import PasswordService, TokenService
from services.users import UserService

security = HTTPBearer()
LoginFormDep = Annotated[OAuth2PasswordRequestForm, Depends()]
AuthHeaderDep = Annotated[HTTPAuthorizationCredentials, Depends(security)]

class AuthService:
    def __init__(self, session: DBSessionDep):
        self.password_service = PasswordService()
        self.token_service = TokenService(session)
        self.user_service = UserService(session)

    async def register_user(self, user: UserCreate) -> User:
        if await self.user_service.get(user.username):
            raise AlreadyRegisteredException
        hashed_password = await self.password_service.get_password_hash(
            user.password
        )
        return await self.user_service.create(
            UserInDB(username=user.username, hashed_password=hashed_password)
        )

    async def authenticate_user(self, form_data: LoginFormDep) -> Token: ![1](assets/1.png)
        if not (user := await self.user_service.get_user(form_data.username)):
            raise UnauthorizedException
        if not await self.password_service.verify_password(
            form_data.password, user.hashed_password
        ):
            raise UnauthorizedException
        return await self.token_service.create_access_token(user._asdict())

    async def get_current_user(self, credentials: AuthHeaderDep) -> User:
        if credentials.scheme != "Bearer":
            raise UnauthorizedException
        if not (token := credentials.credentials):
            raise UnauthorizedException
        payload = self.token_service.decode(token)
        if not await self.token_service.validate(payload.get("sub")):
            raise UnauthorizedException
        if not (username := payload.get("username")):
            raise UnauthorizedException
        if not (user := await self.user_service.get(username)):
            raise UnauthorizedException
        return user

    async def logout(self, credentials: AuthHeaderDep) -> None:
        payload = self.token_service.decode(credentials.credentials)
        await self.token_service.deactivate(payload.get("sub"))

    # Add Password Reset Method
    async def reset_password(self): ...
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO6-1)

The core authentication logic of the application that verifies whether a user exists and their password credentials. Returns `False` if any checks fail.

You can now use the `AuthService` to register and authenticate users using their credentials. Refer to [Example 8-11](#auth_controllers) to see how the `AuthService` is used to create the required dependencies for a dedicated authentication router.

##### Example 8-11\. Implement authentication controllers to enable login and registration functionality

```py
# routes/auth.py

from typing import Annotated
from entities import User
from fastapi import APIRouter, Depends
from models import TokenOut, UserOut
from services.auth import AuthService

auth_service = AuthService()
RegisterUserDep = Annotated[User, Depends(auth_service.register_user)]
AuthenticateUserCredDep = Annotated[
    str, Depends(auth_service.authenticate_user_with_credentials)
]
AuthenticateUserTokenDep = Annotated[User, Depends(auth_service.register_user)]
PasswordResetDep = Annotated[None, Depends(auth_service.reset_password)] ![1](assets/1.png)

router = APIRouter(prefix="/auth", tags=["Authentication"]) ![2](assets/2.png)

@router.post("/register")
async def register_user_controller(new_user: RegisterUserDep) -> UserOut:
    return new_user

@router.post("/token") ![3](assets/3.png)
async def login_for_access_token_controller(
    access_token: AuthenticateUserCredDep,
) -> TokenOut:
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout", dependencies=[Depends(auth_service.logout)]) ![3](assets/3.png) ![4](assets/4.png)
async def logout_access_token_controller() -> dict:
    return {"message": "Logged out"}

@router.post("reset-password") ![3](assets/3.png)
async def reset_password_controller(credentials: str) -> dict:
    return {
        "message": "If an account exists, "
        "a password reset link will be sent to the provided email"
    }
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO7-1)

Create an instance of the `AuthService` and declare reusable annotated dependencies.

[![2](assets/2.png)](#co_authentication_and_authorization_CO7-2)

Create a separate API router for authentication endpoints.

[![3](assets/3.png)](#co_authentication_and_authorization_CO7-3)

Implement endpoints for registering users, user login (token issuance), user logout (token revocation), and password reset.

[![4](assets/4.png)](#co_authentication_and_authorization_CO7-5)

Since the `LogoutUserDep` dependency won’t return anything, inject it within the dependency array of the router.

Once you have a dedicated authentication router, create a separate resource router to group all your resource endpoints within. With both routers, you can now add them to your FastAPI app, as shown in [Example 8-12](#routers), to complete the JWT authentication work.

##### Example 8-12\. Refactor FastAPI application to use routers

```py
# routes/resource.py

from fastapi import APIRouter

router = APIRouter(prefix="/generate", tags=["Resource"]) ![1](assets/1.png)

@router.get("/generate/text", ...)
def serve_language_model_controller(...):
    ...

@router.get("/generate/audio", ...)
def serve_text_to_audio_model_controller(...)
    ...

... # Add other controllers to the resource router here

# main.py

from typing import Annotated
import routes
from entities import User
from fastapi import Depends, FastAPI
from services.auth import AuthService

auth_service = AuthService()
AuthenticateUserDep = Annotated[User, Depends(auth_service.get_current_user)]

...

app = FastAPI(lifespan=lifespan)

app.include_router(routes.auth.router, prefix="/auth", tags=["Auth"]) ![2](assets/2.png)
app.include_router(
    routes.resource.router,
    dependencies=[AuthenticateUserDep],
    prefix="/generate",
    tags=["Generate"],
) ![3](assets/3.png)
...  # Add other routes to the app here
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO8-1)

Refactor existing endpoints to be grouped under a separate API router named the resource router.

[![2](assets/2.png)](#co_authentication_and_authorization_CO8-2)

Add both auth and resource routers to the FastAPI `app` router.

[![3](assets/3.png)](#co_authentication_and_authorization_CO8-3)

Protect the resource endpoints by injecting the `AuthenticateUserDep` dependency at the router level. Requests must now include an `Authorization` header with a bearer token to be authenticated with the resource router.

Massive congratulations! You now have a fully working GenAI service protected by JWT authentication, which can be deployed to production with some additional work.

In the next section, you’ll learn a few ideas on additional enhancements you can make to the system to tighten the security of your JWT authentication system.

### Authentication flows

You will need to handle several authentication flows to fully implement a usable JWT authentication system.

The *core* authentication flows include the following:

User registration

New users will want to register a new account by providing their emails and a secure password. Your authentication logic may check for password strength, no existing users with the same email, and that the user reconfirms the password and email. You should also avoid storing the user’s raw password in the database.

User login

On each user login, your system can generate, store, and provide a unique temporary access token (i.e., JWT) if a user supplies their correct credentials. Your protected resource server routers should reject any incoming requests that don’t contain a valid JWT. Valid JWTs can be verified through their signature and checked against the valid tokens specified in the database.

User logout

When the user logs out, your system can revoke the currently issued token and prevent future malicious login attempts with the current token.

In addition to the core flows, you should also consider *secondary* flows to implement a production-ready authentication system. These flows could be used for:

Verifying identity

To prevent spambots from registering active accounts in your system and consuming server resources, you will want some form of user verification mechanism in place. For instance, add email verification by integrating an emailing server to your authentication system.

Resetting passwords

Users can forget their passwords at any time. You will want to implement a flow for users to reset their passwords. If a user resets their password, all active tokens in the database against their user account must be revoked.

Forcing logout

Revoke all previously generated access tokens of a user on all clients to prevent stolen tokens from being used to access the system.

Disabling user accounts

Administrators or users may want to disable their accounts to prevent future login attempts.

Deleting user accounts

This is required if users would like to remove their accounts from your systems. Depending on your data storage requirements, you may want to delete personally identifiable information while keeping other associated data.

Blocking successive login attempts

Temporarily disable an account that has had multiple failed login attempts within a short time span.

Providing refresh tokens

Generate both short-lived *access* tokens and long-lived *refresh* tokens. Since access tokens can expire frequently to reduce the window of opportunity for attackers to use a stolen token, clients can reuse their refresh token to request new access tokens. This removes the need for frequent logins while maintaining security of the system against attackers.

Two-factor authentication (2FA) or multifactor authentication (MFA)

You can secure your system against exposed password-protected accounts by requiring 2FA or MFA as an additional protection layer. 2FA/MFA examples include SMS/email verification, one-time passwords (OTPs), or randomly generated number sequences from a paired authentication app as a second login step before an access token can be generated.

###### Warning

The aforementioned list is not exhaustive. You may want to check out [“OWASP Top 10 Web Applications Security Risks”](https://oreil.ly/xAGfn) and [“OWASP Authentication Cheat Sheet”](https://oreil.ly/oSyuz) for the full list of considerations when implementing your own JWT authentication from scratch.

In addition to following the OWASP top 10 guidelines, you should use security mechanisms such as *rate limiting*, *geo/IP-tracking*, and *account lockouts* to defend against various attacks.

You can also consider using third-party authentication providers (such as Okta/Auth0, Firebase Auth, KeyCloak, Amazon Cognito, etc.) that include these security features in their services.

While credentials-based authentication using JWTs can be considered a production-ready authentication system and be further enhanced with MFA systems in place, the mechanism has its own limitations. For instance, as previously mentioned, requiring credentials and storing hashed passwords in a database can retain security risks if attackers leverage password spraying or credential stuffing brute-force attacks.

In addition, if you require access to user resources external to your system, you will need to implement additional mechanisms to verify your application’s identity to external identity providers. Since this remains a common need in many applications and services, a protocol called OAuth has been developed to facilitate the whole process.

Let’s explore how you can use OAuth authentication to add more login options for users and access external user resources. This can enhance the performance of your GenAI services and generate higher-quality outputs.

# Implementing OAuth Authentication

We touched upon the concept of OAuth authentication via identity providers earlier in this chapter.

OAuth is an open standard for access delegation, often used to grant websites or applications limited access to user information without exposing passwords. It allows you to authenticate users using identity providers such as Google, Facebook, etc., and grants your application access to user resources like calendars, files, social feeds, etc., on external services.

By using OAuth, you can simplify the implementation of authentication in your app by leveraging existing identity providers instead of creating your own authentication mechanisms such as JWT.

*Identity providers* (IDPs) are platforms that enable other applications, such as your GenAI service, to integrate with and rely on their identity and authentication systems to access resources on behalf of users via a standardized process. The IDP authenticates users and issues security tokens that assert the user’s identity and other attributes. GitHub, Google, Microsoft 365, Apple, Meta, and LinkedIn are only a handful of hundreds of identity providers.

The protocol powering this entire flow under the hood is *OAuth 2.0*, an authorization framework giving applications limited access to another service on behalf of a user.

Using this approach, your application can redirect users to identity provider platforms so that users can grant limited timed access to their accounts on those platforms. After the user gives consent, your application can perform operations on their behalf on their resources like calendars or read their profile information including personally identifiable information such as emails or images.

As a result, OAuth authentication is often used to verify the identity of users as you trust the external platform/identity provider’s authentication process. Therefore, this approach reduces the burden of storing and securing user credentials in your system, which can be prone to brute-force attacks on weak or compromised passwords.

In this section, you’re going to implement a variant of OAuth based on the *authorization code flow* that’s commonly used in modern applications. The step-by-step process is as follows:

1.  The user clicks the login button in your application to start the authentication flow.

2.  The user is redirected to the identity provider’s login page, and your application supplies a client ID and secret to the identity provider to identify itself.

3.  The user logs into their account and is presented with a consent screen like the one shown in [Figure 8-7](#oauth2_consent_screen) presenting them the scopes (i.e., permissions) that your application is requesting on their behalf.

    ![bgai 0807](assets/bgai_0807.png)

    ###### Figure 8-7\. Example consent screen

4.  The user grants all, some, or none of the requested scopes.

5.  If consent is not rejected by the user (i.e., the resource owner), the identity provider’s authorization server issues your application a *grant code* to an endpoint that you provide called the *redirect URI*. If your redirect URI is not previously approved with the identity provider, the identity provider will reject to issue a grant code here.

6.  After your application receives a grant code associated with the user session, permitted scopes, and your application’s client ID, it can exchange this grant code with the authorization server for a *short-lived access token* and a *longer-lived refresh token*. You can use the refresh token to request new access tokens without having to restart the whole authentication process.

7.  Your application can now use this access token to access the provider’s resource server to perform operations on behalf of the user on their resources. As a result, you can authenticate the user via the identity provider to resources on your system.

###### Tip

Through the OAuth process, the authorization server may also issue a *state* parameter or *CSRF token*, which your application must supply as it communicates with the identity provider’s servers. The purpose of the state parameter or CSRF token is to protect against cross-site request forgery (CSRF) attacks.

With CSRF, attackers may steal an authenticated session to forge authenticated requests to the resource servers without the user’s knowledge.

[Figure 8-8](#oauth2) shows the full OAuth authentication flow.

![bgai 0808](assets/bgai_0808.png)

###### Figure 8-8\. OAuth authentication flow

Now that you have a high-level overview of the OAuth authentication flow, let’s implement it inside FastAPI with an identity provider such as GitHub to fully understand the underlying mechanisms.

## OAuth Authentication with GitHub

The first step to setting up the OAuth authentication is to create a set of client ID and secret credentials within GitHub so that their systems can identify your application.

You can generate a client ID and secret from GitHub by visiting the developer settings under your GitHub profile and creating an OAuth application.^([5](ch08.html#id994))

With your new application client ID and secret, you can now redirect users to the GitHub authorization server from your application by following [Example 8-13](#oauth_redirect).

##### Example 8-13\. Redirect users to the GitHub authorization server to start the OAuth process

```py
# routes/auth.py

import secrets
from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

client_id = "your_client_id"
client_secret = "your_client_secret"

router = APIRouter()

...

@router.get("/oauth/github/login", status_code=status.HTTP_301_REDIRECT)
def oauth_github_login_controller(request: Request) -> RedirectResponse:
    state = secrets.token_urlsafe(16)
    redirect_uri = request.url_for("oauth_github_callback_controller")
    response = RedirectResponse(
        url=f"https://github.com/login/oauth/authorize"
        f"?client_id={client_id}"
        f"&scope=user"
        f"&state={state}"
        f"&redirect_uri={redirect_uri}"
    ) ![1](assets/1.png)
    csrf_token = secrets.token_urlsafe(16)
    request.session["x-csrf-state-token"] = csrf_token
    return response
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO9-1)

Redirect user to the GitHub authorization server to log into their account while supplying your application credentials, a requested scope, and a CSRF state value to prevent against CSRF attacks.

As you can see in [Example 8-13](#oauth_redirect), the scope of the request is the user, meaning that once users log into their GitHub account, they will be presented with a consent screen for your application to be granted access to their user profile.

Now that you have a backend endpoint redirecting requests to the GitHub authorization server, you can put a button in your client-side application to hit this endpoint and start the OAuth process with GitHub (see [Example 8-14](#oauth_client)).

##### Example 8-14\. Adding a GitHub login button to the Streamlit client-side application

```py
# client.py

import requests
import streamlit as st

if st.button("Login with GitHub"):
    response = requests.get("http://localhost:8000/auth/oauth/github/login")
    if not response.ok:
        st.error("Failed to login with GitHub. Please try again later")
        response.raise_for_status()
```

You now have implemented the redirect flow that starts the OAuth authentication process with GitHub as the identity provider.

When users log into their GitHub account, GitHub will show them a consent screen similar to [Figure 8-7](#oauth2_consent_screen).

If the user accepts the consent, GitHub will redirect the user back to your application with a grant code and a state. You should check whether the state matches the previously generated state.

###### Warning

If the states do not match, a third party has made the request, and you should stop the process.

Once you have the grant code, you can send this to the GitHub authorization to exchange it for an access token, as shown in [Example 8-15](#oauth_exchange).

##### Example 8-15\. Exchanging grant code with an access token while protecting against CSRF attacks

```py
# dependencies/auth.py

from typing import Annotated
import aiohttp
from fastapi import Depends, HTTPException
from loguru import logger

client_id = "your_client_id"
client_secret = "your_client_secret"

async def exchange_grant_with_access_token(code: str) -> str:
    try:
        body = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://github.com/login/oauth/access_token",
                json=body,
                headers=headers,
            ) as resp:
                access_token_data = await resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch the access token. Error: {e}")
        raise HTTPException(
            status_code=503, detail="Failed to fetch access token"
        )

    if not access_token_data:
        raise HTTPException(
            status_code=503, detail="Failed to obtain access token"
        )

    return access_token_data.get("access_token", "")

ExchangeCodeTokenDep = Annotated[str, Depends(exchange_grant_with_access_token)]
```

You can now add a new endpoint that accepts requests from the GitHub authorization server. This callback endpoint should have a CSRF protection to guard against third parties impersonating the authorization server. If the request from GitHub is forged, the state parameter provided and the one stored in the request session won’t match.

[Example 8-16](#oauth_callback) shows the callback endpoint implementation.

##### Example 8-16\. Implement callback endpoint to get access token while protecting against CSRF attacks

```py
# routes/auth.py

from dependencies.auth import ExchangeCodeTokenDep
from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

...

def check_csrf_state(request: Request, state: str) -> None:
    if state != request.session.get("x-csrf-token"):
        raise HTTPException(detail="Bad request", status_code=401)

@router.get("/oauth/github/callback", dependencies=[Depends(check_csrf_state)])
async def oauth_github_callback_controller(
    access_token: ExchangeCodeTokenDep,
) -> RedirectResponse:
    response = RedirectResponse(url=f"http://localhost:8501")
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response
```

In [Example 8-16](#oauth_callback), you are using the request session for CSRF protection, but this won’t work without adding the Starlette’s `SessionMiddlware` first to maintain a secure user session that’s only mutable on the server side, as shown in [Example 8-17](#oauth_session).

##### Example 8-17\. Add a session middleware to manage session state for protecting against CSRF attacks

```py
# main.py

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

...

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")
```

###### Warning

Avoid relying on HTTP *cookies* to store and read the `state` between request sessions as cookies can be read and manipulated by third parties. Never trust any data that comes from the client.

By writing the unique `state` to the session in [Example 8-13](#oauth_redirect) and comparing it with the `state` value in the incoming request query parameters, you can then confirm the identity of the request.

In this case, the requester is the GitHub authorization server sending you a grant `code`. Once you receive the grant `code`, you then exchange it with the GitHub authorization server for an access token.

###### Tip

The process shown in the OAuth-related code examples can also be implemented with the open source `authlib` package for simpler implementation, as the package handles most of the work for you.

Finally, you can use the access token you received from the authorization server to fetch user information such as their name, email, and profile image to register their identity in your application.

[Example 8-18](#oauth_user_info) demonstrates how to implement an endpoint that returns the user info from GitHub if the request supplies an access token as part of the request’s authorization header.

###### Warning

Ideally, you should avoid sharing the user’s GitHub access token with the user’s browser. If the token is stolen, your application is responsible for compromising the user’s GitHub account.

Instead, create and share your own short-lived access token tied to the GitHub access token to authenticate the user with your application. If your application token is stolen, you avoid compromising user accounts beyond the scope of your application.

##### Example 8-18\. Use access token to get user information from GitHub resource servers

```py
# routes/auth.py

from typing import Annotated
import aiohttp
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()
HTTPBearerDep = Annotated[HTTPAuthorizationCredentials, Depends(security)]

...

async def get_user_info(credentials: HTTPBearerDep) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {credentials.credentials}"}
            async with session.get(
                "https://api.github.com/user", headers=headers
            ) as resp:
                return await resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to obtain user info - Error: {e}"
        )

GetUserInfoDep = Annotated[dict, Depends(get_user_info)]

@router.get("/oauth/github/callback")
async def get_current_user_controller(user_info: GetUserInfoDep) -> dict:
    return user_info
```

Congratulations! You should now have a working authentication system that leverages OAuth2 to authenticate users.

## OAuth2 Flow Types

The OAuth2 flow that you just implemented was the *authorization code flow* (ACF). However, there are other flows that you can choose depending on the use case. The identity provider documentation may present you with solutions for various flows, which can feel overwhelming if you’re not aware of these use cases.

### Authorization code flow

The authorization code flow is the common approach for applications that leverage servers and backend APIs such as FastAPI, using code grants to issue access tokens.

A more secure variant of ACF leverages *proof key for exchange* (PKCE, pronounced “pixie”). You can use the ACF-PKCE flow where you cannot protect the authorization code from being stolen—for instance, on mobile devices.

During the ACF-PKCE flow, you add a hashed secret called the `code_challenge` when sending the initial request to the identity provider. Then you present the unhashed secret `code_verifier` again to exchange the authorization code for an access token.

In essence, PKCE protects against *authorization code interception* attacks—shown in [Figure 8-10](#oauth_interception_attack)—by adding a layer of verification during the token exchange process.

![bgai 0810](assets/bgai_0810.png)

###### Figure 8-10\. OAuth2 authorization code interception attack on a mobile device

### Implicit flow

For single-page applications (SPAs) where there is no separate backend, you can also use the *implicit flow*, which skips the authorization grant code to directly get an access token. Implicit flow is less secure than the previous flow but enhances the user experience.

### Client credentials flow

If you’re building a backend service for machine-to-machine communication and where no browsers will be involved, then you can use the *client credentials flow*. Here you can exchange your client ID and secret for an access token to access your own resources on the identity provider servers (i.e., you won’t have access on behalf of other users).

### Resource owner password credentials flow

*Resource owner password credentials flow* is like the client credentials flow but uses a username and password of the user to get an access token. As the credentials are being exchanged directly with the authorization server, you should avoid using this flow as much as possible.

### Device authorization flow

Finally, there is the *device authorization flow* that’s mostly used for devices with limited input capabilities, such as when you log into your Apple TV account on your smart TV by scanning a QR code and using a web browser from your phone.

[Table 8-2](#oauth_flows_comparison) compares the various flows to help you select the right option for your own use case, based on your specific requirements and constraints.

Table 8-2\. Comparison of OAuth2 authorization flows

| Flow | Description | Considerations | Use cases |
| --- | --- | --- | --- |
| Authorization code flow (including PKCE) | Get the authorization code via a user login and exchange for an access token. |  
*   Provider’s access token must be securely stored and never exposed to the browser.

*   Use ACF-PKCE flow if possible for enhanced security.

 |  
*   Server-side applications and web applications with a backend server that can securely handle the client secret and access tokens.

*   Mobile applications if using a PKCE token as client credentials can’t be securely stored.

 |
| Implicit flow | Get an access token without an authorization code. |  
*   Less secure as the access token is exposed to the browser.

*   Use when using the authorization code flow is not possible.

 | Single-page applications (SPAs) where user experience is prioritized over security, when prototyping, or where the authorization code flow is not possible. |
| Client credentials flow | The client directly exchanges its client credentials (client ID and client secret) for an access token. | No user interaction involved, meant for scenarios where the client is acting on its own behalf. Ensure secure storage of client credentials. | Server-to-server applications. |
| Resource owner password credentials flow | Exchange user’s username and password directly for an access token. |  
*   High security risk as user’s credentials are handled directly.

*   Only use in legacy systems where other flows are not supported.

 | Legacy applications or highly trusted environments. |
| Device authorization flow | Visit a URL on another device to enter a code for an access token. | Requires a second device with a web browser for the user to authenticate. | Devices with limited input capabilities, like smart TVs, gaming consoles, or IoT devices. |

You should now feel more confident in securing your application with a variety of commercial identity verification mechanisms, including the various OAuth2 flows that leverage external IDPs.

Authentication forms the first step to securing your services by identifying who the users of your system are.

A question remains: what should happen when a user is logged into your services? Can they fetch data, interact with models, and mutate resources as they please, or would you rather control their interactions in your services?

These are problems that an authorization system will tackle, which we will talk about next.

# Authorization

So far, we’ve been covering various authentication mechanisms including the basic, token-based (JWT), and OAuth2 for securing your applications.

As mentioned earlier, authentication systems identify and verify actors, whereas the authorization systems enforce *permissions* in an application (i.e., who can do what on which resource).

In this section, you’re going to learn about the authorization system that takes into account the following:

*   The *actor* (i.e., the user or a third-party service acting on behalf of the user)

*   The *action* being undertaken

*   The impact of the action on *resources*

In essence, an authorization system can be compared to a function that accepts three inputs—*actor*, *action*, *resource*—and returns a *Boolean decision* to *allow* or *deny* a request. To implement the authorization function, you will require *authorization data* such as user attributes, relationships (like team/group/org memberships), resource ownership, roles, and permissions passed through a set of *abstract rules* to determine the Boolean allow/deny decisions.

Once a decision is made, you can *enforce* the authorization by either allowing actions (such as fetching or mutating resources) or denying requests (such as sending 403 Forbidden responses, redirecting users, hiding resources, locking accounts, etc.).

On the surface level, implementing authorization can be simple. Using a few conditional statements, you can check whether a user has permissions to perform an action. However, this naive approach can get complex to manage as the number of places you need to implement authorization steps increases. This issue becomes worse as you make changes to the logic across the application, making the system complex and adding finer controls. You may end up duplicating logic or making future changes more difficult, and the authorization rules may deeply be interwoven in your application logic, making separation from the rest of the application more challenging.

In such cases, authorization models can be useful to help you navigate the complexity of managing authorization decisions and enforcements in your applications.

## Authorization Models

There are a few common *authorization models* that you can learn to make structuring and implementing an authorization system easier:

Role-based access control (RBAC)

Authorization is based on the roles assigned to users, where each role has specific permissions. For instance, administrators can access every available GenAI model, bypassing authorization rules enforced on users.

Relationship-based access control (ReBAC)

Authorization is determined by the relationships between entities, such as user-to-user (i.e., follower, friend, connection) or user-to-resource (i.e., group, team, org) relationships. For instance, this could authorize a user who is a member of a team to access premium models purchased by that team.

Attribute-based access control (ABAC)

Authorization decisions are made based on attributes of users, resources, and the environment, allowing for fine-grained access control. For instance, a conversation with a *public* attribute is viewable by everyone, and a user with a *paid* attribute can access premium GenAI models.

RBAC is the simplest authorization model but won’t provide the enhanced granular controls and flexibility of other authorization models. ABAC controls provide more fine-grained access control and can override both ReBAC and RBAC rules. Furthermore, ReBAC can also override or extend RBAC controls.

[Table 8-4](#authorization_methods_comparison) compares the three authorization models.

Table 8-4\. Comparison of authorization methods

| Type | Benefits | Limitations | Use cases |
| --- | --- | --- | --- |
| Role-based (RBAC) | Simplifies management | Limited flexibility | Enterprise environments, access control, financial systems, healthcare systems |
| Relationship-based (ReBAC) | Fine-grained control | Needs relationship data from various sources with complex permission evaluations | Social networks, collaborative platforms, content-sharing applications, project management tools |
| Attribute-based (ABAC) | Highly flexible | Needs attribute data from various sources with complex permission evaluations | Dynamic environments, cloud services, IoT systems, regulatory compliance, personalized user experiences |

These three authorization models also have a hierarchical relationship, as demonstrated in [Figure 8-11](#authorization_models).

![bgai 0811](assets/bgai_0811.png)

###### Figure 8-11\. Authorization models

Let’s now discuss each authorization model in detail, starting with the RBAC model.

## Role-Based Access Control

Using *roles* is a widely adopted model for implementing authorization in applications due to their simplicity.

Roles are straightforward to understand. They normally correspond to whom the user is and what they want to do in the application. Sometimes authorization roles can directly map to roles in your organization’s hierarchy.

You can group permissions under a role that can then be *assigned* to users to grant user those permissions. A *permission* specifies the action that a user can take on resources, such as if the user can interact with the paid LLM model provided by your service.

For better administrative and user experience, you can create multiple roles with preset permissions to reduce decision fatigue when setting user permissions. Instead of having to set a vast number of permissions, you can assign a few predefined roles.

A common starting point for many commercial services is user and administrator roles. While a member can access the core functionality of the application such as interacting with GenAI models, and reading and writing resources, they won’t be able to view data of other users or manage roles. On the other hand, administrators can assign and remove roles, view and mutate every resource, or disable and enable accounts. Administrators may also have access to early features such as GenAI models that normal users can’t access yet, as shown in [Figure 8-12](#rbac_example).

![bgai 0812](assets/bgai_0812.png)

###### Figure 8-12\. RBAC example where only administrators have access to image-based GenAI models

You can implement a simple RBAC authorization model to control the GenAI services that your users can access, as shown in [Example 8-19](#rbac).

##### Example 8-19\. Implementing RBAC using FastAPI dependencies

```py
# dependencies/auth.py

from entities import User
from fastapi import Depends, HTTPException, status
from services.auth import AuthService

async def is_admin(user: User = Depends(AuthService.get_current_user)) -> User: ![1](assets/1.png)
    if user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not allowed to perform this action",
        )
    return user

# routers/resource.py

from dependencies.auth import is_admin
from fastapi import APIRouter, Depends
from services.auth import AuthService

router = APIRouter(
    dependencies=[Depends(AuthService.get_current_user)], ![3](assets/3.png)
    prefix="/generate",
    tags=["Resource"],
)

@router.post("/image", dependencies=[Depends(is_admin)])
async def generate_image_controller(): ![2](assets/2.png)
    ...

@router.post("/text") ![3](assets/3.png)
async def generate_text_controller():
    ...
```

[![1](assets/1.png)](#co_authentication_and_authorization_CO10-1)

Implement the `is_admin` authorization dependency guard on top of the `Auth​Ser⁠vice.get_current_user` dependency. Mark the function as `async` since the child dependency is performing an async operation against the database.

[![2](assets/2.png)](#co_authentication_and_authorization_CO10-3)

Use the authorization guard dependency to deny access to the image generation service for nonadmin authenticated users.

[![3](assets/3.png)](#co_authentication_and_authorization_CO10-2)

Nonadmin authenticated users can still access other resource controllers since the router is secured by an authentication guard dependency.

Using the same logic shown in [Example 8-19](#rbac), you can construct varying system prompt templates or use different model variants fine-tuned to each role.

###### Warning

Bear in mind that implementing authorization at the application layer is more secure than delegating it to the GenAI model. LLMs and other GenAI models can be vulnerable to *prompt injection* attacks where an attacker manipulates the input to the model to bypass system instructions to produce unauthorized and harmful outputs.

Future versions of LLMs and other GenAI models may mitigate prompt injection risks by enforcing custom authorization rules internally using extensions like the *control neural network (ControlNet)* in Stable Diffusion models.

To create more complex RBAC authorization logic than the one shown in [Example 8-19](#rbac), you can implement *subdependencies* or an *abstract dependency*. Both approaches will leverage FastAPI’s powerful *hierarchical dependency graphs* as authorization guards to enforce permissions in your GenAI service.

As an example, if you add new roles in the future that inherit a subset of permissions of another role (i.e., moderators and admins), then you can follow either of the approaches shown in [Figure 8-13](#complex_rbac_approaches).

![bgai 0813](assets/bgai_0813.png)

###### Figure 8-13\. Approaches for implementing complex RBAC models

You can implement complex RBAC authorization logic using abstract dependencies, as shown in [Example 8-20](#complex_rbac_abstract).

##### Example 8-20\. Implementing complex RBAC authorization using abstract dependencies

```py
# dependencies/auth.py

from typing import Annotated

from entities import User
from fastapi import APIRouter, Depends, HTTPException, status
from services.auth import AuthService

CurrentUserDep = Annotated[User, Depends(AuthService.get_current_user)]

async def has_role(user: CurrentUserDep, roles: list[str]) -> User:
    if user.role not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not allowed to perform this action",
        )
    return user

# routes/resource.py

...

@router.post(
    "/image",
    dependencies=[Depends(lambda user: has_role(user, ["ADMIN", "MODERATOR"]))],
)
async def generate_image_controller():
    ...

@router.post(
    "/text", dependencies=[Depends(lambda user: has_role(user, ["EDITOR"]))]
)
async def generate_text_controller():
    ...
```

In summary, RBAC simplifies permission management by assigning permissions to roles rather than individuals, making it easier to manage and audit. It is scalable and efficient for organizations with well-defined roles and responsibilities.

However, RBAC can lead to role explosion when many granular roles are necessary, making it hard to manage. It also lacks the flexibility to handle complex hierarchical relationships like teams and groups alongside setting dynamic permissions based on attributes like user preferences, time, and privacy settings, which limits its granularity compared to ReBAC or ABAC.

## Relationship-Based Access Control

*Relationship-based access control* is an extension of RBAC with a focus on relationships between resources and users.

With this mode, instead of just setting roles at the user level across the entire application, you must set roles and permissions at the resource level. This means you will have to confirm the actions each role can take on every resource type. For example, instead of assigning a “moderator” role to a user that grants access to all resources (i.e., conversations, teams, users, etc.), you would assign specific permissions to the moderator role for each resource. A moderator might have read and delete permissions on the conversation resource but only read permission on the team resource.

This model allows you to create authorization policies based on hierarchical and nested structures within your data and be visualized as graphs where nodes can be represented as resources/identities and edges as relationships.

Since you can create authorization rules based on relationships, this can save you lots of time setting permissions at an instance level. As an example, instead of sharing every private LLM conversation in your app one by one, you can group them under a team or a folder and share the folder or add members to the team instead. In ReBAC, children instances can inherit parent’s permissions, as shown in [Figure 8-14](#rebac_example). It’s the same for related instances if needed.

![bgai 0814](assets/bgai_0814.png)

###### Figure 8-14\. Example ReBAC where a user can see the team’s private conversations and threads

The example shown in [Figure 8-14](#rebac_example) demonstrates both organization and hierarchical relationships between users (i.e., teams and members) and resources (conversations and threads).

###### Tip

If you decide to adopt the ReBAC model, I recommend visually mapping out the relationships between resources and identities in your application.

This work includes mapping out *policies* (i.e., rules), *resources* and available *actions* on them, *resource-level roles*, and *relationships* between entities.

A big problem that ReBAC solves by extending RBAC is the explosion of roles within the RBAC model by combining relationships with roles. It is ideal for managing permissions in complex hierarchical structures and allows for reverse queries, enabling efficient permission definitions using teams and groups. However, ReBAC can be complex to implement and maintain, resource-intensive, difficult to audit, and not as fine-grained as ABAC for dynamic permissions based on attributes like time or location.

## Attribute-Based Access Control

*Attribute-based access control* authorization model expands basic RBAC roles by setting access control rules based on *conditions applied to attributes* to implement more granular policies. As an example, ABAC can prevent users from uploading sensitive documents into your RAG-enabled services if the document contains *personally identifiable information (PII)* (i.e., `upload.has_pii=true`).

Another example of ABAC can be seen in SaaS applications like ChatGPT where only paid users have access to the service’s premium GenAI models (see [Figure 8-15](#abac_example)).

![bgai 0815](assets/bgai_0815.png)

###### Figure 8-15\. ABAC example where only paid users have access to premium GenAI models

Since the freedom to set policies based on attributes is infinite, the ABAC model allows for significantly fine-grained authorization policies. However, ABAC can be cumbersome for managing hierarchical structures, making it challenging to determine which users have access to a specific resource. For example, if you have a policy that grants access based on attributes like user role, data sensitivity level, and project membership, determining all users who can access a specific dataset requires evaluating these attributes for every user.

While less complicated than ReBAC, ABAC can still be challenging to implement, in particular in large and complex applications that support a large number of roles, users, and attributes.

## Hybrid Authorization Models

If you’ve worked with larger applications in the past, you will notice that they combine features of the RBAC, ReBAC, and ABAC authorization models. For instance, administrators may have access to any resource and user management/authentication features (RBAC), and users can share their private resources by setting visibility attribute to `public` (ABAC) and can add members to their team for collaborating on private resources.

A hybrid approach combining RBAC, ReBAC, and ABAC models may give you the strengths of all the authorization models:

*   RBAC simplifies permission management by assigning roles to users, making it easy to manage and audit.

*   ReBAC is perfect for managing hierarchical relationships and reverse queries, making it suitable for complex hierarchical structures.

*   ABAC provides fine-grained control based on user and resource attributes, allowing for dynamic and context-aware permissions.

[Figure 8-16](#authorization_hybrid) demonstrates the hybrid authorization model.

![bgai 0816](assets/bgai_0816.png)

###### Figure 8-16\. Hybrid authorization model based on roles, relationships, and attributes

To implement the hybrid authorization combining RBAC, ReBAC, and ABAC models, you can follow [Example 8-21](#rbac_rebac_abac).

##### Example 8-21\. Implementing the hybrid authorization model combining RBAC, ReBAC, and ABAC

```py
# dependencies/auth.py

from typing import Annotated
from fastapi import Depends, HTTPException, status
...  # import services and entities here

CurrentUserDep = Annotated[User, Depends(AuthService.get_current_user)]
TeamMembershipRep = Annotated[Team, Depends(TeamService.get_current_team)]
ResourceDep = Annotated[Resource, Depends(ResourceService.get_resource)]

def authorize(
    user: CurrentUserDep, resource: ResourceDep, team: TeamMembershipRep
) -> bool:
    if user.role == "ADMIN":
        return True
    if user.id in team.members:
        return True
    if resource.is_public:
        return True
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Access Denied"
    )

# routes/resource.py

from dependencies.auth import authorize
from fastapi import APIRouter, Depends

router = APIRouter(
    dependencies=[Depends(authorize)], prefix="/generate", tags=["Resource"]
)

@router.post("/image")
async def generate_image_controller(): ...

@router.post("/text")
async def generate_text_controller(): ...
```

As you define rules and permissions based on each authorization model, you also may decide to bypass rules if certain conditions are met. This can lead to complex logic and create a maintenance burden on your application code.

Since implementing a hybrid model can be complex, you may consider developing a separate authorization service to eliminate the need for significant code changes with volatile permissions that change frequently.

Using an external system for authorization decisions allows your application’s authorization logic to remain consistent, as shown in [Figure 8-17](#authz_separate).

![bgai 0817](assets/bgai_0817.png)

###### Figure 8-17\. Separating the authorization service from the GenAI service

[Example 8-22](#authorization_separate_example) shows how to develop a separate authorization system.

##### Example 8-22\. Using an authorization service with the GenAI service

```py
# authorization_api.py (Authorization Service)

from typing import Annotated, Literal
from fastapi import Depends, FastAPI
from pydantic import BaseModel

...  # import services and entities here

CurrentUserDep = Annotated[User, Depends(AuthService.get_current_user)]
ActionRep = Annotated[Literal["READ", "CREATE", "UPDATE", "DELETE"], str]
ResourceDep = Annotated[Resource, Depends(ResourceService.get_resource)]

class AuthorizationResponse(BaseModel):
    allowed: bool

app = FastAPI()

app.get("/authorize")
def authorization_controller(
    user: CurrentUserDep, resource: ResourceDep, action: ActionRep
) -> AuthorizationResponse:
    if user.role == "ADMIN":
        return AuthorizationResponse(allowed=True)
    if action in user.permissions.get(resource.id, []):
        return AuthorizationResponse(allowed=True)
    ...  # Other permission checks
    return AuthorizationResponse(allowed=False)

# genai_api.py (GenAI Service)

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

class AuthorizationData(BaseModel):
    user_id: int
    resource_id: int
    action: str

authorization_client = ...  # Create authorization client

async def enforce(data: AuthorizationData) -> bool:
    response = await authorization_client.decide(data)
    if response.allowed:
        return True
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Access Denied"
    )

router = APIRouter(
    dependencies=[Depends(enforce)], prefix="/generate", tags=["Resource"]
)

@router.post("/text")
async def generate_text_controller():
    ...
```

As you can see in [Example 8-22](#authorization_separate_example), using an external system to authorize user actions on resources helps you and your team to modularize authorization logic with more complex and volatile permission requirements.

However, since developing a complex external authorization service from scratch can take a lot of your time, you may want to consider using authorization providers (such as Oso, Permify, Okta/Auth0, etc.) with your authentication.

# Summary

In this chapter, you learned about both authentication and authorization mechanisms to secure your GenAI services.

Earlier in the chapter, you were introduced to several authentication methods, including basic, token-based, OAuth, and key-based authentication. To gain hands-on experience, you implemented several authentication systems from scratch in your FastAPI service, which helped you understand the underlying mechanisms. This included managing user passwords, creating and using JWT access tokens, and implementing authentication flows for user verification. Additionally, you learned how to integrate your services with identity providers like GitHub using the OAuth2 standard to authenticate users and access external user resources in your application.

While you were building the authentication system, you also learned about attack vectors such as credential stuffing, password spraying, cross-site request forgery, open redirect, and phishing attacks.

Furthermore, you explored authorization systems that determine and enforce access levels based on authorization data and logic. You learned how authorization systems can become complex and how different models, including RBAC, ReBAC, and ABAC, can assist in managing permissions in your applications.

In the next chapter, you will focus on testing, including writing unit, integration, end-to-end, and regression tests. You’ll be introduced to concepts like testing boundaries, coverage, mocking, patching, parameterization, isolation, and idempotency, which will help you write maintainable and effective tests as your applications grow in complexity. Specifically, you’ll learn how to test GenAI services that use probabilistic models and interface with asynchronous systems.

^([1](ch08.html#id970-marker)) Open Worldwide Application Security Project is an online community that produces resources on system software and web application security.

^([2](ch08.html#id971-marker)) Key-based authentication won’t be discussed further as it involves complex cryptographic principles that are beyond the scope of this chapter.

^([3](ch08.html#id972-marker)) In a *timing attack*, attackers try to guess passwords by comparing and analyzing elapsed password evaluation times with the password length. Therefore, to prevent timing attacks, cryptographic algorithms must check passwords within a constant time span.

^([4](ch08.html#id982-marker)) Typical salt lengths include 16 bytes (128 bits) for balancing performance and security, or 32 bytes (256 bits) for securing sensitive systems. You can use cryptographic libraries such as `passlib` to generate these salts correctly.

^([5](ch08.html#id994-marker)) For up-to-date instructions, please visit [the GitHub documentation](https://oreil.ly/tWg6w).