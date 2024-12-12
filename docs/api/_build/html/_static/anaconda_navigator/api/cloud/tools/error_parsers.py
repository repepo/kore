# -*- coding: utf-8 -*-

"""
Utilities for reusable and conditional error processing.

This utilities allow converting this:

.. code-block:: python

    try:
        ...
    except ValueError as exception:
        result = fix_value_error(exception)
    except KeyError as exception:
        result = fix_key_error(exception)
    except requests.RequestsException as exception:
        if exception.response is not None:
            if (exception.response.status_code == 409) and (exception.response.json()['error']['code'] = 'some_exc'):
                result = fix_duplicate_environment(exception)
            elif (exception.response.status_code == 422):
                result = fix_unprocessable_entity(exception)
            else:
                result = fix_default(exception)
        else:
            result = fix_default(exception)
    except BaseException as exception:
        result = fix_default(exception)

to this:

.. code-block:: python

    handlers = HttpErrorHandlers()
    handlers.register_handler(BaseException, fix_default)
    handlers.register_handler(ValueError, fix_value_error)
    handlers.register_handler(KeyError, fix_key_error)
    handlers.register_http_handler(fix_duplicate_environment, 409, 'some_exc')
    handlers.register_http_handler(fix_unprocessable_entity, 422)

    try:
        ...
    except BaseException as exception:
        result = handlers.handle(exception)

.. rubric:: Handlers

Handlers are used to process exceptions. Each handler is associated with particular exception and exception context.

If you have multiple handlers for inherited exceptions - the closest one to the exception will be selected. E.g. you
have handlers for :code:`KeyError`, :code:`Exception` and :code:`BaseException`. For handling :code:`LookupError` will
be used handler for :code:`Exception`. For :code:`KeyboardInterrupt` - handler for :code:`BaseException`. And for any
"direct hit" - corresponding handler.

Exception context - is additional sequence of details, which can be used to select a more specific handlers. E.g. for
:exc:`~requests.RequestException` this context have next form:
:code:`[response.status_code, response.json()['error']['code']]`. If the :code:`response.json()['error']['code']` is
unavailable - this part of context is omitted.

Contexts are matched in a "longest prefix" manner. E.g. you have handlers for the same error, but with contexts
:code:`['a', 'b', 'c']`, :code:`['a', 'b', 'd']`, :code:`['a', 'b']`, :code:`['a', 'e']`. If you have exception with a
context :code:`['a', 'b', 'c', 'x']` - handler for context :code:`['a', 'b', 'c']` will be used. For exception with a
context :code:`['a', 'b']` - handler for context :code:`['a', 'b']` will be used. For exception with an empty context -
none of the above will be used.

This exception contexts are detected for each exception separately with exception parsers (see
:meth:`ErrorHandlers.register_parser`).

Functions to handle exceptions must have next format:

.. code-block:: python

    def handler(exception: BaseException) -> typing.Any:
        ...

Result value of the handler will be returned through the :meth:`ErrorHandlers.handle`.
"""

__all__ = ['ErrorHandlers', 'HttpErrorHandlers']

import typing
import requests

ExceptionT_contra = typing.TypeVar('ExceptionT_contra', bound=BaseException, contravariant=True)


class Handler(typing.Protocol[ExceptionT_contra]):  # pylint: disable=too-few-public-methods
    """Common protocol for error handling functions."""

    def __call__(self, exception: ExceptionT_contra) -> typing.Any:
        """
        Execute handler.

        :param exception: Exception to handle.
        :return: Result of the exception handling.
        """


class Parser(typing.Protocol[ExceptionT_contra]):  # pylint: disable=too-few-public-methods
    """Common protocol for context parsers for exceptions."""

    def __call__(self, exception: ExceptionT_contra) -> typing.Tuple[typing.Any, ...]:
        """
        Prepare exception context.

        :param exception: Exception, for which context must be prepared.
        :return: Context for the exception.
        """


class ErrorHandler(typing.NamedTuple):
    """
    Internal description of the error handler.

    .. py:attribute:: exception_type

        Type of the exception, that can be handled bu this handler.

    .. py:attribute:: context

        Context prefix required to execute this handler.

    .. py:attribute:: handler

        Function to handle an exception.
    """

    exception_type: typing.Type[BaseException]
    context: typing.Tuple[typing.Any, ...]
    handler: 'Handler'


class ErrorParser(typing.NamedTuple):
    """
    Internal description of the error parser.

    .. py:property:: exception_type

        Type of the exception, that can be handled bu this parser.

    .. py:property:: parser

        Function to prepare context for exception.
    """

    exception_type: typing.Type[BaseException]
    parser: 'Parser'


class ErrorHandlers:
    """Collection of error handlers."""

    __slots__ = ('__handlers', '__parsers')

    def __init__(self) -> None:
        """Initialize new :class:`~ErrorHandlers` instance."""
        self.__handlers: typing.Final[typing.List[ErrorHandler]] = []
        self.__parsers: typing.Final[typing.List[ErrorParser]] = []

    def handle(self, exception: BaseException) -> typing.Any:
        """
        Find handler for the `exception` and execute it.

        :param exception: Exception to handle.
        :return: Whatever handler returns.
        """
        context: typing.Tuple[typing.Any, ...] = ()

        parser: ErrorParser
        for parser in self.__parsers:
            if isinstance(exception, parser.exception_type):
                context = parser.parser(exception)
                break

        handler: ErrorHandler
        for handler in self.__handlers:
            if isinstance(exception, handler.exception_type) and (handler.context == context[:len(handler.context)]):
                return handler.handler(exception)

        return None

    def register_handler(
            self,
            exception_type: typing.Type[ExceptionT_contra],
            handler: 'Handler[ExceptionT_contra]',
            *context: typing.Any,
    ) -> None:
        """
        Add new exception handler.

        :param exception_type: Exception, which may be handled.
        :param handler: Handler function.
        :param context: Required context prefix.
        """
        index: int = 0
        while index < len(self.__handlers):
            current: ErrorHandler = self.__handlers[index]
            if issubclass(exception_type, current.exception_type):
                if current.exception_type != exception_type:
                    break
                if len(current.context) <= len(context):
                    break
            index += 1

        self.__handlers.insert(index, ErrorHandler(exception_type=exception_type, context=context, handler=handler))

    def register_parser(
            self,
            exception_type: typing.Type[ExceptionT_contra],
            parser: 'Parser[ExceptionT_contra]',
    ) -> None:
        """
        Add new exception parser.

        :param exception_type: Exceptions, which may be parsed.
        :param parser: Context parsing function.
        """
        index: int = 0
        while index < len(self.__parsers):
            if issubclass(exception_type, self.__parsers[index].exception_type):
                break
            index += 1

        self.__parsers.insert(index, ErrorParser(exception_type=exception_type, parser=parser))


class HttpErrorHandlers(ErrorHandlers):
    """Collection of error handlers, which is prepared to parse HTTP requests from CloudAPI."""

    def __init__(self) -> None:
        """Initialize new :class:`~HttpErrorHandlers` instance."""
        super().__init__()
        self.register_parser(requests.RequestException, self.parse_http)

    @staticmethod
    def parse_http(exception: requests.RequestException) -> typing.Tuple[typing.Any, ...]:
        """Retrieve context for :class:`~requests.RequestException`."""
        result: typing.List[typing.Any] = []

        if exception.response is not None:
            result.append(exception.response.status_code)
            try:
                result.append(exception.response.json()['error']['code'])
            except (ValueError, TypeError, KeyError):
                pass

        return tuple(result)

    def register_http_handler(self, handler: 'Handler[requests.RequestException]', *context: typing.Any) -> None:
        """
        Add new exception handler for :class:`~requests.RequestException`.

        Interface is similar to :meth:`~ErrorHandlers.register_handler`, except missing `exception_type` argument.
        """
        self.register_handler(requests.RequestException, handler, *context)
