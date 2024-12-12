# Copyright (c) 2022, Riverbank Computing Limited
# All rights reserved.
#
# This copy of SIP is licensed for use under the terms of the SIP License
# Agreement.  See the file LICENSE for more details.
#
# This copy of SIP may also used under the terms of the GNU General Public
# License v2 or v3 as published by the Free Software Foundation which can be
# found in the files LICENSE-GPL2 and LICENSE-GPL3 included in this package.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# The builtin Python exceptions across all supported versions of Python.
PYTHON_EXCEPTIONS = (
    'BaseException',
    'Exception',
    'StopIteration',
    'GeneratorExit',
    'ArithmeticError',
    'LookupError',
    'StandardError',                # Python v2.

    'AssertionError',
    'AttributeError',
    'BufferError',
    'EOFError',
    'FloatingPointError',
    'OSError',
    'ImportError',
    'IndexError',
    'KeyError',
    'KeyboardInterrupt',
    'MemoryError',
    'NameError',
    'OverflowError',
    'RuntimeError',
    'NotImplementedError',
    'SyntaxError',
    'IndentationError',
    'TabError',
    'ReferenceError',
    'SystemError',
    'SystemExit',
    'TypeError',
    'UnboundLocalError',
    'UnicodeError',
    'UnicodeEncodeError',
    'UnicodeDecodeError',
    'UnicodeTranslateError',
    'ValueError',
    'ZeroDivisionError',
    'EnvironmentError',             # Python v2.
    'IOError',                      # Python v2.
    'WindowsError',                 # Python v2.
    'VMSError',                     # Python v2.

    'BlockingIOError',
    'BrokenPipeError',
    'ChildProcessError',
    'ConnectionError',
    'ConnectionAbortedError',
    'ConnectionRefusedError',
    'ConnectionResetError',
    'FileExistsError',
    'FileNotFoundError',
    'InterruptedError',
    'IsADirectoryError',
    'NotADirectoryError',
    'PermissionError',
    'ProcessLookupError',
    'TimeoutError',

    'Warning',
    'UserWarning',
    'DeprecationWarning',
    'PendingDeprecationWarning',
    'SyntaxWarning',
    'RuntimeWarning',
    'FutureWarning',
    'ImportWarning',
    'UnicodeWarning',
    'BytesWarning',
    'ResourceWarning',
)