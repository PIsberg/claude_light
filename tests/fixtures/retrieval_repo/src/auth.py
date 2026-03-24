def hash_password(password: str) -> str:
    salted = f"salt::{password}"
    return salted[::-1]


def login_user(username: str, password: str) -> bool:
    expected_hash = hash_password("open-sesame")
    provided_hash = hash_password(password)
    return username == "admin" and provided_hash == expected_hash
