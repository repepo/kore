# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Workers and manager for running long processes in threads without blocking GUI.
"""

from collections import deque
import contextlib
import os

from qtpy.QtCore import QByteArray, QObject, QProcess, QThread, QTimer, Signal  # pylint: disable=no-name-in-module

from anaconda_navigator.utils.py3compat import PY2, to_text_string


WIN = os.name == 'nt'


def handle_qbytearray(obj, encoding):
    """Qt/Python2/3 compatibility helper."""
    if isinstance(obj, QByteArray):
        obj = obj.data()

    return to_text_string(obj, encoding=encoding)


class PythonWorker(QObject):
    """
    Generic python worker for running python code on threads.

    For running processes (via QProcess) use the ProcessWorker.
    """
    sig_started = Signal(object)
    sig_partial = Signal(object, object, object)  # worker, stdout, stderr
    sig_finished = Signal(object, object, object)  # worker, stdout, stderr

    def __init__(self, func, args, kwargs):
        """Generic python worker for running python code on threads."""
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_finished = False
        self._started = False

    def is_finished(self):
        """Return True if worker status is finished otherwise return False."""
        return self._is_finished

    def start(self):
        """Start the worker (emits sig_started signal with worker as arg)."""
        if not self._started:
            self.sig_started.emit(self)
            self._started = True

    def terminate(self):
        """Mark the worker as finished."""
        self._is_finished = True

    def _start(self):
        """Start process worker for given method args and kwargs."""
        error = None
        output = None

        try:
            output = self.func(*self.args, **self.kwargs)
        except Exception as err:  # pylint: disable=broad-except
            error = err

        if not self._is_finished:
            self.sig_finished.emit(self, output, error)
        self._is_finished = True


class DummyWorker(QObject):  # pylint: disable=too-few-public-methods
    """Process worker based on a QProcess for non blocking UI."""

    sig_started = Signal(object)
    sig_partial = Signal(object, object, object)
    sig_finished = Signal(object, object, object)


class ProcessWorker(QObject):  # pylint: disable=too-many-instance-attributes
    """Process worker based on a QProcess for non blocking UI."""

    sig_started = Signal(object)
    sig_partial = Signal(object, object, object)
    sig_finished = Signal(object, object, object)

    def __init__(self, cmd_list, environ=None):
        """
        Process worker based on a QProcess for non blocking UI.

        Parameters
        ----------
        cmd_list : list of str
            Command line arguments to execute.
        environ : dict
            Process environment,
        """
        super().__init__()
        self._result = None
        self._cmd_list = cmd_list
        self._fired = False
        self._communicate_first = False
        self._partial_stdout = None
        self._partial_stderr = None
        self._started = False

        self._timer = QTimer()
        self._process = QProcess()
        self._set_environment(environ)

        self._timer.setInterval(150)
        self._timer.timeout.connect(self._communicate)
        self._process.readyReadStandardOutput.connect(self._partial)

    @staticmethod
    def _get_encoding():
        """Return the encoding/codepage to use."""
        enco = 'utf-8'

        #  Currently only cp1252 is allowed?
        if WIN:
            import ctypes  # pylint: disable=import-outside-toplevel
            codepage = to_text_string(ctypes.cdll.kernel32.GetACP())
            enco = 'cp' + codepage

        return enco

    def _set_environment(self, environ):
        """Set the environment on the QProcess."""
        if environ:
            q_environ = self._process.processEnvironment()
            for k, v in environ.items():
                q_environ.insert(k, v)
            self._process.setProcessEnvironment(q_environ)

    def _partial(self):
        """Callback for partial output."""
        raw_stdout = self._process.readAllStandardOutput()
        stdout = handle_qbytearray(raw_stdout, self._get_encoding())

        if self._partial_stdout is None:
            self._partial_stdout = stdout
        else:
            self._partial_stdout += stdout

        # NOTE: use the piece or the cummulative?
        self.sig_partial.emit(self, stdout, None)

    def _communicate(self):
        """Callback for communicate."""
        if (not self._communicate_first and
                self._process.state() == QProcess.NotRunning):
            self.communicate()
        elif self._fired:
            self._timer.stop()

    def communicate(self):
        """Retrieve information."""
        self._communicate_first = True
        self._process.waitForFinished()

        enco = self._get_encoding()
        if self._partial_stdout is None:
            raw_stdout = self._process.readAllStandardOutput()
            stdout = handle_qbytearray(raw_stdout, enco)
        else:
            stdout = self._partial_stdout

        if self._partial_stderr is None:
            raw_stderr = self._process.readAllStandardError()
            stderr = handle_qbytearray(raw_stderr, enco)
        else:
            stderr = self._partial_stderr

        if PY2:
            stdout = stdout.decode()
            stderr = stderr.decode()

        result = [stdout, stderr]
        self._result = result

        if not self._fired:
            self.sig_finished.emit(self, result[0], result[-1])

        self._fired = True

        return result

    def close(self):
        """Close the running process."""
        self._process.close()

    def is_finished(self):
        """Return True if worker has finished processing."""
        return self._process.state() == QProcess.NotRunning and self._fired

    def _start(self):
        """Start process."""
        if not self._fired:
            # print(self._cmd_list)
            self._partial_ouput = None  # pylint: disable=attribute-defined-outside-init
            if self._cmd_list:
                self._process.start(self._cmd_list[0], self._cmd_list[1:])
            self._timer.start()

    def terminate(self):
        """Terminate running processes."""
        if self._process.state() == QProcess.Running:
            with contextlib.suppress(BaseException):
                self._process.terminate()
        self._fired = True

    def write(self, data):  # pylint: disable=missing-function-docstring
        if self._started:
            self._process.write(data)

    def start(self):
        """Start worker."""
        if not self._started:
            self.sig_started.emit(self)
            self._started = True


class WorkerManager(QObject):  # pylint: disable=too-many-instance-attributes
    """Spyder Worker Manager for Generic Workers."""

    def __init__(self, max_threads=10):
        """Spyder Worker Manager for Generic Workers."""
        super().__init__()
        self._queue = deque()
        self._queue_workers = deque()
        self._threads = []
        self._workers = []
        self._timer = QTimer()
        self._timer_worker_delete = QTimer()
        self._running_threads = 0
        self._max_threads = max_threads

        # Keeps references to old workers
        # Needed to avoid C++/python object errors
        self._bag_collector = deque()

        self._timer.setInterval(333)
        self._timer.timeout.connect(self._start)
        self._timer_worker_delete.setInterval(5000)
        self._timer_worker_delete.timeout.connect(self._clean_workers)

    def _clean_workers(self):
        """Dereference workers in workers bag periodically."""
        while self._bag_collector:
            self._bag_collector.popleft()
        self._timer_worker_delete.stop()

    def _start(self, worker=None):
        """Start threads and check for inactive workers."""
        if worker:
            self._queue_workers.append(worker)

        if self._queue_workers and self._running_threads < self._max_threads:
            self._running_threads += 1
            worker = self._queue_workers.popleft()
            thread = QThread()
            if isinstance(worker, PythonWorker):
                worker.moveToThread(thread)
                worker.sig_finished.connect(thread.quit)
                thread.started.connect(worker._start)  # pylint: disable=protected-access
                thread.start()
            elif isinstance(worker, ProcessWorker):
                thread.quit()
                worker._start()  # pylint: disable=protected-access
            self._threads.append(thread)
        else:
            self._timer.start()

        if self._workers:
            for w in self._workers:
                if w.is_finished():
                    self._bag_collector.append(w)
                    self._workers.remove(w)

        if self._threads:
            for t in self._threads:
                if t.isFinished():
                    self._threads.remove(t)
                    self._running_threads -= 1

        if len(self._threads) == 0 and len(self._workers) == 0:
            self._timer.stop()
            self._timer_worker_delete.start()

    def create_python_worker(self, func, *args, **kwargs):
        """Create a new python worker instance."""
        worker = PythonWorker(func, args, kwargs)
        self._create_worker(worker)
        return worker

    def create_process_worker(self, cmd_list, environ=None):
        """Create a new process worker instance."""
        worker = ProcessWorker(cmd_list, environ=environ)
        self._create_worker(worker)
        return worker

    def _create_worker(self, worker):
        """Common worker setup."""
        worker.sig_started.connect(self._start)
        self._workers.append(worker)
