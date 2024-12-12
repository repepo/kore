class AuthenticationError(Exception):
    pass


class InvalidTokenError(AuthenticationError):
    pass


class TokenNotFoundError(Exception):
    pass


class LoginRequiredError(Exception):
    pass


class TokenExpiredError(Exception):
    pass
