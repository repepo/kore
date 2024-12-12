# -*- coding: utf-8 -*-

"""
Utilities for streaming encoding of JSON objects.

This component is created to minimize memory usage when creating a large JSON files by adding ability to stream content.
"""

__all__ = ['JsonWriter']

import abc
import decimal
import typing


STRING_ESCAPES: typing.Final[typing.Mapping[str, str]] = {
    '\"': '\\"',
    '\\': '\\\\',
    '/': '\\/',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
}


class JsonSerializer(metaclass=abc.ABCMeta):
    """
    Common base for data serializers.

    Instances of this class are used to export JSON-compatible objects to the files.

    :param types: Collection of types, which can be serialized by this serializer.
    """

    __slots__ = ('__types',)

    def __init__(self, *types: typing.Type[typing.Any]) -> None:
        """Initialize new :class:`~JsonSerializer` instance."""
        self.__types: typing.Final[typing.Tuple[typing.Type[typing.Any], ...]] = types

    @property
    def types(self) -> typing.Tuple[typing.Type[typing.Any], ...]:  # noqa: D401
        """Collection of types, that are supported by this serializer."""
        return self.__types

    @abc.abstractmethod
    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""


class KeywordSerializer(JsonSerializer):
    """Serializer for keywords."""

    def __init__(self) -> None:
        """Initialize new :class:`~KeywordWriter` instance."""
        super().__init__(bool, type(None))

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        if instance is True:
            context.write('true')
        if instance is False:
            context.write('false')
        if instance is None:
            context.write('null')


class StringSerializer(JsonSerializer):
    """Serializer for string values."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize new :class:`~StringWriter` instance."""
        super().__init__(str)

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        instance = typing.cast(str, instance)

        character: str
        context.write('"')
        for character in instance:
            context.write(STRING_ESCAPES.get(character, character))
        context.write('"')


class NumberSerializer(JsonSerializer):
    """Serializer for number values."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize new :class:`~NumberWriter` instance."""
        super().__init__(int, float, decimal.Decimal)

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        context.write(str(instance))


class MappingSerializer(JsonSerializer):
    """Serializer for mappings."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize new :class:`~MappingWriter` instance."""
        super().__init__(typing.Mapping)

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        instance = typing.cast(typing.Mapping[str, typing.Any], instance)

        key: str
        value: typing.Any
        initial: bool = True
        context.write('{')
        for key, value in instance.items():
            if not isinstance(key, str):
                raise TypeError('key of mapping must be a str instance')
            if initial:
                initial = False
            else:
                context.write(',')
            context.serialize(key)
            context.write(':')
            context.serialize(value)
        context.write('}')


class SequenceSerializer(JsonSerializer):
    """Serializer for sequences."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize new :class:`~SequenceWriter` instance."""
        super().__init__(typing.Sequence)

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        instance = typing.cast(typing.Sequence[typing.Any], instance)

        value: typing.Any
        initial: bool = True
        context.write('[')
        for value in instance:
            if initial:
                initial = False
            else:
                context.write(',')
            context.serialize(value)
        context.write(']')


class TextSerializer(JsonSerializer):
    """
    Serializer for TextIO instances.

    This serializer copies text from a file into JSON string.
    """

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize new :class:`~TextWriter` instance."""
        super().__init__(object)

    def serialize(self, context: 'JsonContext', instance: typing.Any) -> None:
        """Convert `instance` to a JSON-compatible value."""
        instance = typing.cast(typing.TextIO, instance)

        buffer: str
        buffer_size: int = context.configuration.get('buffer_size', 8192)
        context.write('"')
        while True:
            buffer = instance.read(buffer_size)
            if not buffer:
                break

            character: str
            for character in buffer:
                context.write(STRING_ESCAPES.get(character, character))
        context.write('"')


class JsonContext:
    """
    Context for a single serialized JSON.

    This class is intended only for internal use. For public interface - use :class:`~JsonWriter`.
    """

    __slots__ = ('__configuration', '__serializers', '__state', '__stream')

    def __init__(
            self,
            configuration: typing.Mapping[str, typing.Any],
            serializers: typing.Sequence[JsonSerializer],
            stream: typing.TextIO,
    ) -> None:
        """Initialize new :class:`~JsonContext` instance."""
        self.__configuration: typing.Final[typing.Mapping[str, typing.Any]] = configuration
        self.__serializers: typing.Final[typing.Sequence[JsonSerializer]] = serializers
        self.__state: typing.Final[typing.MutableMapping[str, typing.Any]] = {}
        self.__stream: typing.Final[typing.TextIO] = stream

    @property
    def configuration(self) -> typing.Mapping[str, typing.Any]:  # noqa: D401
        """Collection of common configuration values."""
        return self.__configuration

    @property
    def state(self) -> typing.MutableMapping[str, typing.Any]:  # noqa: D401
        """Storage for serialization state values, that might be shared between serializers."""
        return self.__state

    def close(self) -> None:
        """Close target `stream`."""
        self.__stream.close()

    def serialize(self, instance: typing.Any) -> None:
        """Serialize and write instance to the target `stream`."""
        serializer: JsonSerializer
        for serializer in self.__serializers:
            if isinstance(instance, serializer.types):
                break
        else:
            raise TypeError(f'can not serialize {type(instance).__name__} instance')

        serializer.serialize(context=self, instance=instance)

    def write(self, value: str) -> None:
        """Write string to target `stream`."""
        self.__stream.write(value)


class JsonWriter:
    """
    JSON serializer, optimized for streaming sources and targets.

    :param target: Target file to write JSON to. Might be both TextIO or path to file.
    :param serializers: Collection of custom serializers (for customization or extended support).
    :param configuration: Additional options to modify serialization behavior.
    """

    __slots__ = ('__context',)

    def __init__(
            self,
            target: typing.Union[str, typing.TextIO],
            *,
            serializers: typing.Optional[typing.Sequence[JsonSerializer]] = None,
            **configuration: typing.Any,
    ) -> None:
        """Initialize new :class:`~JsonWriter` instance."""
        if isinstance(target, str):
            target = open(target, 'wt', encoding='utf-8')  # pylint: disable=consider-using-with
        if serializers is None:
            serializers = (
                KeywordSerializer(),
                StringSerializer(),
                NumberSerializer(),
                MappingSerializer(),
                SequenceSerializer(),
                TextSerializer(),
            )

        self.__context: typing.Final[JsonContext] = JsonContext(
            configuration=configuration,
            serializers=serializers,
            stream=target,
        )

    @property
    def configuration(self) -> typing.Mapping[str, typing.Any]:  # noqa: D401
        """Collection of common configuration values."""
        return self.__context.configuration

    def close(self) -> None:
        """Close `target`."""
        self.__context.close()

    def serialize(self, instance: typing.Any) -> None:
        """Serialize and write instance to the `target`."""
        self.__context.serialize(instance)
