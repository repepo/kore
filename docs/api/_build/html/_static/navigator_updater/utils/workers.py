# -*- coding: utf-8 -*-

"""Integration with worker framework to make API calls asynchronous."""

from __future__ import annotations

__all__ = [
    'TaskCanceledError', 'Call', 'Duration',
    'TaskStatus', 'TaskResult', 'TaskWorker', 'TaskThread',
    'TaskContext', 'TaskModifier', 'InsertArgument', 'Task', 'CancelContext', 'AddCancelContext',
]

import abc
import contextlib
import datetime
import enum
import typing
from qtpy import QtCore
from . import singletons


ACTION = typing.Callable[[], typing.Any]
FUNCTION = typing.Callable[..., typing.Any]


class TaskCanceledError(Exception):
    """Task was canceled during its workflow."""

    def __str__(self) -> str:
        """Prepare string representation of the instance."""
        return 'Task was cancelled during its execution'


class Call:
    """
    Signature of a function call.

    :param function: Function to call.
    :param args: Positional arguments for the `function`.
    :param kwargs: Keyword arguments for the `function`.
    """

    __slots__ = ('__function', '__args', '__kwargs')

    def __init__(
            self,
            function: FUNCTION,
            args: typing.Optional[typing.Sequence[typing.Any]] = None,
            kwargs: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    ) -> None:
        """Initialize new :class:`~Call` instance."""
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        self.__function: typing.Final[FUNCTION] = function
        self.__args: typing.Final[typing.Sequence[typing.Any]] = args
        self.__kwargs: typing.Final[typing.Mapping[str, typing.Any]] = kwargs

    @property
    def function(self) -> FUNCTION:  # noqa: D401
        """Function to call."""
        return self.__function

    @property
    def args(self) -> typing.Sequence[typing.Any]:  # noqa: D401
        """Positional arguments to call `function` with."""
        return self.__args

    @property
    def kwargs(self) -> typing.Mapping[str, typing.Any]:  # noqa: D401
        """Keyword arguments to call `function` with."""
        return self.__kwargs

    def __call__(self) -> typing.Any:
        """Execute stored function with both positional and keyword arguments."""
        return self.__function(*self.__args, **self.__kwargs)


class Duration:
    """
    Simple helper to track execution duration.

    :param start: Optional start of the tracked time span.

                  If it is not provided - current timestamp will be used.

    :param stop: Optional end of the tracked time span.

                 Might be omitted on initialization and then set with :meth:`~Duration.finish`.
    """

    __slots__ = ('__start', '__stop')

    def __init__(
            self,
            start: typing.Optional[datetime.datetime] = None,
            stop: typing.Optional[datetime.datetime] = None,
    ) -> None:
        """Initialize new :class:`~Duration` instance."""
        if start is None:
            start = datetime.datetime.now()

        self.__start: typing.Final[datetime.datetime] = start
        self.__stop: typing.Optional[datetime.datetime] = stop

    @property
    def start(self) -> datetime.datetime:  # noqa: D401
        """Start of the tracked time span."""
        return self.__start

    @property
    def stop(self) -> typing.Optional[datetime.datetime]:  # noqa: D401
        """Stop of the tracked time span."""
        return self.__stop

    @property
    def total(self) -> typing.Optional[float]:  # noqa: D401
        """Total number of seconds in this time span."""
        if self.__stop is None:
            return None
        return (self.__stop - self.__start).total_seconds()

    def finish(self, stop: typing.Optional[datetime.datetime] = None) -> None:
        """
        Set the stop value for this duration.

        Might be set only once (either with this method or on initialization).

        :param stop: Value to set stop to.

                     If it is not provided - current timestamp will be used.
        """
        if self.__stop is not None:
            raise TypeError('Duration is already closed')
        if stop is None:
            stop = datetime.datetime.now()
        self.__stop = stop


class TaskStatus(enum.Enum):
    """Possible states of the :class:`~Task` execution."""

    SUCCEEDED = enum.auto()
    CANCELED = enum.auto()
    FAILED = enum.auto()


class TaskResult:
    """
    Details of task execution.

    Instances of this class are provided through the task signals.

    :param call: Signature of the function that was called.
    :param status: Resulted status of the execution.
    :param result: Function output.

                   Must be an exact value for succeeded executions, and exception for failed executions.
    :param duration: Optional information about task execution duration.
    """

    __slots__ = ('__call', '__status', '__result', '__duration')

    def __init__(
            self,
            call: Call,
            status: TaskStatus,
            result: typing.Any = None,
            duration: typing.Optional[Duration] = None,
    ) -> None:
        """Initialize new :class:`~TaskResult` instance."""
        if duration is None:
            duration = Duration()
            duration.finish()

        if status == TaskStatus.CANCELED:
            if result is None:
                result = TaskCanceledError()
            elif not isinstance(result, TaskCanceledError):
                raise TypeError('result must be a TaskCanceledError instance if task is failed')
        elif (status == TaskStatus.FAILED) and (not isinstance(result, BaseException)):
            raise TypeError('result must be an exception if task is failed')

        self.__call: typing.Final[Call] = call
        self.__status: typing.Final[TaskStatus] = status
        self.__result: typing.Final[typing.Any] = result
        self.__duration: typing.Final[Duration] = duration

    @property
    def call(self) -> Call:  # noqa: D401
        """Signature of the function that was called."""
        return self.__call

    @property
    def duration(self) -> Duration:  # noqa: D401
        """Task execution duration."""
        return self.__duration

    @property
    def exception(self) -> typing.Optional[BaseException]:  # noqa: D401
        """Exception, that was raised by the function."""
        if self.__status == TaskStatus.SUCCEEDED:
            return None
        return self.__result

    @property
    def result(self) -> typing.Any:  # noqa: D401
        """
        Function output.

        If function raised an error - this property will also raise the same error.
        """
        if self.__status == TaskStatus.SUCCEEDED:
            return self.__result
        raise self.__result

    @property
    def status(self) -> TaskStatus:  # noqa: D401
        """Resulted status of the execution."""
        return self.__status


class TaskSignals(QtCore.QObject):  # pylint: disable=too-few-public-methods
    """
    Collection of signals for :class:`~TaskWorker`.

    .. py:attribute:: sig_start

        Task started it's execution.

    .. py:attribute:: sig_succeeded

        Task finished its execution successfully.

    .. py:attribute:: sig_canceled

        Task was canceled during it's execution.

    .. py:attribute:: sig_failed

        Task raised an error.

    .. py:attribute:: sig_done

        Task finished its execution.

        This signal is called regardless of execution result.

    """

    sig_start = QtCore.Signal()

    sig_succeeded = QtCore.Signal(TaskResult)
    sig_canceled = QtCore.Signal(TaskResult)
    sig_failed = QtCore.Signal(TaskResult)

    sig_done = QtCore.Signal(TaskResult)


class TaskWorker(QtCore.QRunnable):
    """
    Executor for tasks.

    :param call: Signature of the function to call with the worker.
    :param cancel: Optional cancel function.

                   This function might be called to change `function` execution context, so it would be aware that it
                   should be canceled.

                   If it is not provided - `function` execution will still be finished gracefully, but result will
                   contain only information about cancelled function call, unless it will raise an exception.
    """

    def __init__(self, call: Call, cancel: typing.Optional[ACTION] = None) -> None:
        """Initialize new :class:`~TaskWorker` instance."""
        super().__init__()

        self.__signals: typing.Final[TaskSignals] = TaskSignals()
        self.__call: typing.Final[Call] = call
        self.__cancel: typing.Final[typing.Optional[ACTION]] = cancel
        self.__canceled: bool = False
        self.__result: typing.Optional[TaskResult] = None

    @property
    def call(self) -> Call:  # noqa: D401
        """Signature of the function that is executed in this task."""
        return self.__call

    @property
    def signals(self) -> TaskSignals:  # noqa: D401
        """Collection of signals to track task execution state."""
        return self.__signals

    @property
    def result(self) -> typing.Optional[TaskResult]:  # noqa: D401
        """Task execution result."""
        return self.__result

    def cancel(self) -> None:
        """Cancel execution of current task."""
        if self.__result is not None:
            raise TypeError('Already finished')
        if self.__canceled:
            raise TypeError('Already canceled')

        self.__canceled = True

        if self.__cancel is not None:
            self.__cancel()

    def run(self) -> None:
        """
        Execute task.

        This method is called internally by the Qt.

        :meta private:
        """
        MANAGER.instance._register(self)  # pylint: disable=protected-access
        self.signals.sig_start.emit()

        status: TaskStatus
        response: typing.Any
        duration: typing.Final[Duration] = Duration()
        try:
            if self.__canceled:
                raise TaskCanceledError()
            response = self.__call()
            status = TaskStatus.SUCCEEDED
        except TaskCanceledError as exception:
            response = exception
            status = TaskStatus.CANCELED
        except Exception as exception:  # pylint: disable=broad-except
            response = exception
            status = TaskStatus.FAILED
        finally:
            duration.finish()
            if self.__canceled:
                response = TaskCanceledError()
                status = TaskStatus.CANCELED

        self.__result = TaskResult(
            call=self.__call,
            status=status,
            result=response,
            duration=duration,
        )
        {
            TaskStatus.SUCCEEDED: self.signals.sig_succeeded,
            TaskStatus.CANCELED: self.signals.sig_canceled,
            TaskStatus.FAILED: self.signals.sig_failed,
        }[status].emit(self.__result)
        self.signals.sig_done.emit(self.__result)

    def start(self) -> None:
        """Launch this task."""
        THREAD_POOL.instance.start(self)

    def thread(self) -> 'TaskThread':
        """Start a manageable thread with this task."""
        result: TaskThread = TaskThread(parent=self)
        result.start()
        return result


class TaskThread(QtCore.QThread):
    """
    Wrapper for a :class:`~TaskWorker`, that provides a manageable :class:`~qtpy.QtCore.QThread` interface.

    Should be initialized from :class:`~TaskWorker` instance via :meth:`~TaskWorker.thread`.
    """

    def __init__(self, parent: TaskWorker) -> None:
        """Initialize new :class:`~TaskThread` instance."""
        super().__init__()
        self.__parent: typing.Final[TaskWorker] = parent

    @property
    def call(self) -> Call:  # noqa: D401
        """Signature of the function that is executed in this task."""
        return self.__parent.call

    @property
    def result(self) -> typing.Optional[TaskResult]:  # noqa: D401
        """Task execution result."""
        return self.__parent.result

    def cancel(self) -> None:
        """Cancel execution of current task."""
        self.__parent.cancel()

    def run(self) -> None:
        """
        Execute the thread.

        This method is called internally by the Qt.

        :meta private:
        """
        self.__parent.run()


class TaskContext:
    """
    Context for task signature.

    This is used to prepare contexts for both function :class:`~Call` and/or :class:`~TaskWorker`.

    :param function: Function that should be called.
    :param args: Initial collection of positional arguments to call function with.
    :param kwargs: Initial collection of keyword arguments to call function with.
    :param cancel: Initial value for cancel function, which may be used by :class:`~TaskWorker`.

    Values of `args`, `kwargs` and `cancel` might be modified by :class:`~TaskModifier` instances.
    """

    __slots__ = ('__function', '__args', '__kwargs', '__cancel')

    def __init__(
            self,
            function: FUNCTION,
            args: typing.Optional[typing.Iterable[typing.Any]] = None,
            kwargs: typing.Optional[typing.Mapping[str, typing.Any]] = None,
            cancel: typing.Optional[ACTION] = None,
    ) -> None:
        """Initialize new :class:`~TaskContext` instance."""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        self.__function: typing.Final[FUNCTION] = function
        self.__args: typing.Final[typing.List[typing.Any]] = list(args)
        self.__kwargs: typing.Final[typing.Dict[str, typing.Any]] = dict(kwargs)
        self.__cancel: typing.Optional[ACTION] = cancel

    @property
    def args(self) -> typing.List[typing.Any]:  # noqa: D401
        """Collection of positional arguments to call :attr:`~TaskContext.function` with."""
        return self.__args

    @property
    def call(self) -> Call:  # noqa: D401
        """New :class:`~Call` signature from the collected context."""
        return Call(function=self.function, args=self.args, kwargs=self.kwargs)

    @property
    def cancel(self) -> typing.Optional[ACTION]:  # noqa: D401
        """Function that can cancel function call in a graceful way."""
        return self.__cancel

    @cancel.setter
    def cancel(self, value: typing.Optional[ACTION]) -> None:
        """Update `cancel` value."""
        self.__cancel = value

    @property
    def function(self) -> FUNCTION:  # noqa: D401
        """Function to call from this context."""
        return self.__function

    @property
    def kwargs(self) -> typing.Dict[str, typing.Any]:  # noqa: D401
        """Collection of keyword arguments to call :attr:`~TaskContext.function` with."""
        return self.__kwargs

    @property
    def worker(self) -> TaskWorker:  # noqa: D401
        """New :class:`~TaskWorker` from the collected context."""
        return TaskWorker(call=self.call, cancel=self.cancel)


class TaskModifier(metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """
    Common interface for all :class:`~TaskContext` modifiers.

    This is used to separate context modifiers from the actual function in :class:`~Task`.
    """

    __slots__ = ()

    @abc.abstractmethod
    def apply(self, context: TaskContext) -> None:
        """Apply modification to the :class:`~TaskContext` instance."""


class InsertArgument(TaskModifier):
    """
    Insert new positional argument.

    Places `value` at `target` in :attr:`~TaskContext.args` or :attr:`~TaskContext.kwargs`.

    :param value: Value of the argument to add.
    :param target: Index of the positional or Key of the keyword argument to add.
    """

    __slots__ = ('__value', '__target')

    def __init__(self, value: typing.Any, target: typing.Union[int, str] = 0) -> None:
        """Initialize new :class:`~InsertArgument` instance."""
        self.__value: typing.Final[typing.Any] = value
        self.__target: typing.Final[typing.Union[int, str]] = target

    @property
    def target(self) -> typing.Union[int, str]:  # noqa: D401
        """
        Location for the new argument.

        Might be :code:`int` index for positional arguments, or :code:`str` key for keyword arguments.
        """
        return self.__target

    @property
    def value(self) -> typing.Any:  # noqa: D401
        """Value of the new argument."""
        return self.__value

    def apply(self, context: TaskContext) -> None:
        """Apply modification to the :class:`~TaskContext` instance."""
        if isinstance(self.target, int):
            context.args.insert(self.target, self.value)
        else:
            context.kwargs[self.target] = self.value


# Ideas for extra modifiers:
# - AppendArgument(value) -> args.append(value)
# - EnsureArgument(target, value) -> kwargs.setdefault(target, value)


class Task:
    """
    Decorator for functions and methods to convert them to tasks.

    After decoration - functions and methods might still be called usual way. :meth:`~Task.worker` might be used to
    create a worker for a task.

    :param args: :class:`~TaskModifier` instances to modify a :class:`~TaskContext` before execution.

    .. note::

        Decorator can be called in both :code:`@Task` and :code:`@Task(...)` ways. Both will work.
    """

    __slots__ = ('__function', '__modifiers')

    def __init__(self, *args: typing.Union[FUNCTION, TaskModifier]) -> None:
        """Initialize new :class:`~Task` instance."""
        function: typing.Optional[FUNCTION] = None
        modifiers: typing.List[TaskModifier] = []

        arg: typing.Union[FUNCTION, TaskModifier]
        for arg in args:
            if isinstance(arg, TaskModifier):
                modifiers.append(arg)
            elif function is None:
                function = arg
            else:
                raise TypeError('Task arguments must be TaskModifiers and at most one function')

        self.__function: typing.Optional[FUNCTION] = function
        self.__modifiers: typing.Final[typing.Tuple[TaskModifier, ...]] = tuple(modifiers)

    @property
    def function(self) -> FUNCTION:  # noqa: D401
        """Function that is used for this task."""
        if self.__function is None:
            raise AttributeError('function was not wrapped yet')
        return self.__function

    @property
    def modifiers(self) -> typing.Tuple[TaskModifier, ...]:  # noqa: D401
        """Collection of modifiers for a :class:`~TaskContext`."""
        return self.__modifiers

    def __context(
            self,
            args: typing.Optional[typing.Iterable[typing.Any]] = None,
            kwargs: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    ) -> TaskContext:
        """Prepare a context for execution or converting into a task."""
        modifier: TaskModifier
        result: typing.Final[TaskContext] = TaskContext(function=self.function, args=args, kwargs=kwargs)
        for modifier in self.modifiers:
            modifier.apply(context=result)
        return result

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """
        Execute a function in a usual way.

        .. note::

            This method might be also called when the decorator is called in a :code:`@Task(...)` way to finalize
            function/method wrapping.
        """
        if self.__function is None:
            if (len(args) != 1) or kwargs:
                raise TypeError('function must be the only provided argument for wrapping.')
            self.__function = args[0]
            return self

        return self.__context(args=args, kwargs=kwargs).call()

    def __get__(self, instance: typing.Any, owner: typing.Any) -> 'Task':
        """Prepare a bound task for a methods called for a specific instances."""
        if instance is None:
            return self
        return Task(self.function, *self.modifiers, InsertArgument(instance))

    def worker(self, *args: typing.Any, **kwargs: typing.Any) -> TaskWorker:
        """Prepare worker to execute this task."""
        return self.__context(args=args, kwargs=kwargs).worker


class TaskManager(QtCore.QObject):
    """Central manager for executing tasks."""

    sig_empty = QtCore.Signal()

    sig_start = QtCore.Signal(TaskWorker)
    sig_succeeded = QtCore.Signal(TaskWorker)
    sig_canceled = QtCore.Signal(TaskWorker)
    sig_failed = QtCore.Signal(TaskWorker)
    sig_done = QtCore.Signal(TaskWorker)

    def __init__(self) -> None:
        """Initialize new :class:`~TaskManager` instance."""
        super().__init__()

        self.__cancel: bool = False
        self.__workers: typing.Final[typing.Set[TaskWorker]] = set()

    def __start(self, worker: TaskWorker) -> None:
        """Process start of a worker."""
        self.__workers.add(worker)

        if self.__cancel:
            with contextlib.suppress(TypeError):
                worker.cancel()

        self.sig_start.emit(worker)

    def __succeeded(self, worker: TaskWorker) -> None:
        """Process success of a worker."""
        if worker not in self.__workers:
            return

        self.sig_succeeded.emit(worker)

    def __canceled(self, worker: TaskWorker) -> None:
        """Process cancel of a worker."""
        if worker not in self.__workers:
            return

        self.sig_canceled.emit(worker)

    def __failed(self, worker: TaskWorker) -> None:
        """Process fail of a worker."""
        if worker not in self.__workers:
            return

        self.sig_failed.emit(worker)

    def __done(self, worker: TaskWorker) -> None:
        """Process worker finished execution."""
        try:
            self.__workers.remove(worker)
        except LookupError:
            return

        self.sig_done.emit(worker)

        if not self.__workers:
            self.sig_empty.emit()

    def _register(self, worker: TaskWorker) -> None:
        """Register worker when it starts its execution."""
        if worker in self.__workers:
            return

        self.__start(worker)
        worker.signals.sig_succeeded.connect(lambda _: self.__succeeded(worker))
        worker.signals.sig_canceled.connect(lambda _: self.__canceled(worker))
        worker.signals.sig_failed.connect(lambda _: self.__failed(worker))
        worker.signals.sig_done.connect(lambda _: self.__done(worker))

    def cancel_all(self) -> None:
        """Cancel all running and future workers."""
        self.__cancel = True

        workers: typing.Sequence[TaskWorker] = list(self.__workers)
        if not workers:
            self.sig_empty.emit()
            return

        worker: TaskWorker
        for worker in workers:
            with contextlib.suppress(TypeError):
                worker.cancel()

    def reset(self) -> None:
        """
        Reset task processing.

        This might be useful to allow executing new tasks after calling :meth:`~TaskManager.cancel_all`.
        """
        self.__cancel = False


class CancelContext:
    """
    Simplistic context with a canceled flag.

    :param canceled: Initial canceled state.
    """

    __slots__ = ('__canceled',)

    def __init__(self, canceled: bool = False) -> None:
        """Initialize new :class:`~CancelContext` instance."""
        self.__canceled: bool = canceled

    @property
    def canceled(self) -> bool:  # noqa: D401
        """Task was canceled."""
        return self.__canceled

    def abort_if_canceled(self) -> None:
        """Break execution of a task if it was canceled."""
        if self.__canceled:
            raise TaskCanceledError()

    def cancel(self) -> None:
        """Cancel task execution."""
        self.__canceled = True


class AddCancelContext(TaskModifier):
    """
    Add new :class:`~CancelContext` to the :class:`~TaskContext`.

    This adds new :class:`~CancelContext` instance to the function kwargs and modifies :attr:`~TaskContext.cancel` to
    use the context.

    :param argument: Name of the keyword argument to add context to.

    .. warning::

        Context will be provided if only it was not provided before (with a regular keyword argument).

        Otherwise - it will reuse provided value.
    """

    __slots__ = ('__argument',)

    def __init__(self, argument: str = 'context') -> None:
        """Initialize new :class:`~AddCancelContext` instance."""
        self.__argument: typing.Final[str] = argument

    @property
    def argument(self) -> str:  # noqa: D401
        """Name of the argument to which context will be added."""
        return self.__argument

    def apply(self, context: TaskContext) -> None:
        """Apply modification to the :class:`~TaskContext` instance."""
        if self.argument not in context.kwargs:
            context.kwargs[self.argument] = CancelContext()
        context.cancel = context.kwargs[self.argument].cancel


class ThreadPoolSingleton(singletons.Singleton[QtCore.QThreadPool]):  # pylint: disable=too-few-public-methods
    """Singleton instance of a :class:`~qtpy.QtCore.QThreadPool`."""

    __slots__ = ()

    def _prepare(self) -> QtCore.QThreadPool:
        """Initialize singleton instance."""
        return QtCore.QThreadPool()

    def _release(self) -> None:
        """Destroy singleton instance when :meth:`~Singleton.reset` is called."""
        self.instance.clear()
        self.instance.waitForDone()


MANAGER: typing.Final[singletons.Singleton[TaskManager]] = singletons.SingleInstanceOf(TaskManager)
THREAD_POOL: typing.Final[singletons.Singleton[QtCore.QThreadPool]] = ThreadPoolSingleton()
