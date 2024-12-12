"""
conda_build utils we need, without configuring logging as an import side effect.
"""

import contextlib
import hashlib
import logging
import os
import shutil
import subprocess
import time
from os import stat
from os.path import isdir, isfile, islink

import filelock
from conda.exports import root_dir

log = logging.getLogger(__name__)

# this import misconfigures logging as a side effect
# from conda_build.utils import (
#     ensure_list,
#     get_lock,
#     merge_or_update_dict,
#     move_with_fallback,
#     try_acquire_locks,
# )

string_types = (str,)  # Python 3


class LockError(Exception):
    """Raised when we failed to acquire a lock."""


def ensure_list(arg, include_dict=True):
    """
    Ensure the object is a list. If not return it in a list.

    :param arg: Object to ensure is a list
    :type arg: any
    :param include_dict: Whether to treat `dict` as a `list`
    :type include_dict: bool, optional
    :return: `arg` as a `list`
    :rtype: list
    """
    if arg is None:
        return []
    elif islist(arg, include_dict=include_dict):
        return list(arg)
    else:
        return [arg]


def islist(arg, uniform=False, include_dict=True):
    """
    Check whether `arg` is a `list`. Optionally determine whether the list elements
    are all uniform.

    When checking for generic uniformity (`uniform=True`) we check to see if all
    elements are of the first element's type (`type(arg[0]) == type(arg[1])`). For
    any other kinds of uniformity checks are desired provide a uniformity function:

    .. code-block:: pycon
        # uniformity function checking if elements are str and not empty
        >>> truthy_str = lambda e: isinstance(e, str) and e
        >>> islist(["foo", "bar"], uniform=truthy_str)
        True
        >>> islist(["", "bar"], uniform=truthy_str)
        False
        >>> islist([0, "bar"], uniform=truthy_str)
        False

    .. note::
        Testing for uniformity will consume generators.

    :param arg: Object to ensure is a `list`
    :type arg: any
    :param uniform: Whether to check for uniform or uniformity function
    :type uniform: bool, function, optional
    :param include_dict: Whether to treat `dict` as a `list`
    :type include_dict: bool, optional
    :return: Whether `arg` is a `list`
    :rtype: bool
    """
    if isinstance(arg, str) or not hasattr(arg, "__iter__"):
        # str and non-iterables are not lists
        return False
    elif not include_dict and isinstance(arg, dict):
        # do not treat dict as a list
        return False
    elif not uniform:
        # short circuit for non-uniformity
        return True

    # NOTE: not checking for Falsy arg since arg may be a generator

    if uniform is True:
        arg = iter(arg)
        try:
            etype = type(next(arg))
        except StopIteration:
            # StopIteration: list is empty, an empty list is still uniform
            return True
        # check for explicit type match, do not allow the ambiguity of isinstance
        def uniform(e):
            return type(e) == etype

    try:
        return all(uniform(e) for e in arg)
    except (ValueError, TypeError):
        # ValueError, TypeError: uniform function failed
        return False


# purpose here is that we want *one* lock per location on disk.  It can be
# locked or unlocked at any time, but the lock within this process should all be
# tied to the same tracking mechanism.
_lock_folders = (
    os.path.join(root_dir, "locks"),
    os.path.expanduser(os.path.join("~", ".conda_build_locks")),
)


def get_lock(folder, timeout=900):
    fl = None
    try:
        location = os.path.abspath(os.path.normpath(folder))
    except OSError:
        location = folder
    b_location = location
    if hasattr(b_location, "encode"):
        b_location = b_location.encode()

    # Hash the entire filename to avoid collisions.
    lock_filename = hashlib.sha256(b_location).hexdigest()

    for locks_dir in _lock_folders:
        try:
            if not os.path.isdir(locks_dir):
                os.makedirs(locks_dir)
            lock_file = os.path.join(locks_dir, lock_filename)
            with open(lock_file, "w") as f:
                f.write("")
            fl = filelock.FileLock(lock_file, timeout)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(
            "Could not write locks folder to either system location ({})"
            "or user location ({}).  Aborting.".format(*_lock_folders)
        )
    return fl


def _equivalent(base_value, value, path):
    equivalent = value == base_value
    if isinstance(value, string_types) and isinstance(base_value, string_types):
        if not os.path.isabs(base_value):
            base_value = os.path.abspath(
                os.path.normpath(os.path.join(path, base_value))
            )
        if not os.path.isabs(value):
            value = os.path.abspath(os.path.normpath(os.path.join(path, value)))
        equivalent |= base_value == value
    return equivalent


def merge_or_update_dict(
    base, new, path="", merge=True, raise_on_clobber=False, add_missing_keys=True
):
    if base == new:
        return base

    for key, value in new.items():
        if key in base or add_missing_keys:
            base_value = base.get(key, value)
            if hasattr(value, "keys"):
                base_value = merge_or_update_dict(
                    base_value, value, path, merge, raise_on_clobber=raise_on_clobber
                )
                base[key] = base_value
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                if merge:
                    if base_value != value:
                        try:
                            base_value.extend(value)
                        except (TypeError, AttributeError):
                            base_value = value
                    try:
                        base[key] = list(base_value)
                    except TypeError:
                        base[key] = base_value
                else:
                    base[key] = value
            else:
                if (
                    base_value
                    and merge
                    and not _equivalent(base_value, value, path)
                    and raise_on_clobber
                ):
                    log.debug(
                        "clobbering key {} (original value {}) with value {}".format(
                            key, base_value, value
                        )
                    )
                if value is None and key in base:
                    del base[key]
                else:
                    base[key] = value
    return base


@contextlib.contextmanager
def try_acquire_locks(locks, timeout):
    """Try to acquire all locks.

    If any lock can't be immediately acquired, free all locks.
    If the timeout is reached withou acquiring all locks, free all locks and raise.

    http://stackoverflow.com/questions/9814008/multiple-mutex-locking-strategies-and-why-libraries-dont-use-address-comparison
    """
    t = time.time()
    while time.time() - t < timeout:
        # Continuously try to acquire all locks.
        # By passing a short timeout to each individual lock, we give other
        # processes that might be trying to acquire the same locks (and may
        # already hold some of them) a chance to the remaining locks - and
        # hopefully subsequently release them.
        try:
            for lock in locks:
                lock.acquire(timeout=0.1)
        except filelock.Timeout:
            # If we failed to acquire a lock, it is important to release all
            # locks we may have already acquired, to avoid wedging multiple
            # processes that try to acquire the same set of locks.
            # That is, we want to avoid a situation where processes 1 and 2 try
            # to acquire locks A and B, and proc 1 holds lock A while proc 2
            # holds lock B.
            for lock in locks:
                lock.release()
        else:
            break
    else:
        # If we reach this point, we weren't able to acquire all locks within
        # the specified timeout. We shouldn't be holding any locks anymore at
        # this point, so we just raise an exception.
        raise LockError("Failed to acquire all locks")

    try:
        yield
    finally:
        for lock in locks:
            lock.release()


# with each of these, we are copying less metadata.  This seems to be necessary
#   to cope with some shared filesystems with some virtual machine setups.
#  See https://github.com/conda/conda-build/issues/1426
def _copy_with_shell_fallback(src, dst):
    is_copied = False
    for func in (shutil.copy2, shutil.copy, shutil.copyfile):
        try:
            func(src, dst)
            is_copied = True
            break
        except (OSError, PermissionError):
            continue
    if not is_copied:
        try:
            subprocess.check_call(
                f"cp -a {src} {dst}",
                shell=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            if not os.path.isfile(dst):
                raise OSError(f"Failed to copy {src} to {dst}.  Error was: {e}")


# http://stackoverflow.com/a/22331852/1170370
def copytree(src, dst, symlinks=False, ignore=None, dry_run=False):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]

    # do not copy lock files
    if ".conda_lock" in lst:
        lst.remove(".conda_lock")

    dst_lst = [os.path.join(dst, item) for item in lst]

    if not dry_run:
        for idx, item in enumerate(lst):
            s = os.path.join(src, item)
            d = dst_lst[idx]
            if symlinks and os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass  # lchmod not available
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                _copy_with_shell_fallback(s, d)

    return dst_lst


def merge_tree(
    src, dst, symlinks=False, timeout=900, lock=None, locking=True, clobber=False
):
    """
    Merge src into dst recursively by copying all files from src into dst.
    Return a list of all files copied.

    Like copytree(src, dst), but raises an error if merging the two trees
    would overwrite any files.
    """
    dst = os.path.normpath(os.path.normcase(dst))
    src = os.path.normpath(os.path.normcase(src))
    assert not dst.startswith(src), (
        "Can't merge/copy source into subdirectory of itself.  "
        "Please create separate spaces for these things.\n"
        "  src: {}\n"
        "  dst: {}".format(src, dst)
    )

    new_files = copytree(src, dst, symlinks=symlinks, dry_run=True)
    existing = [f for f in new_files if isfile(f)]

    if existing and not clobber:
        raise OSError(
            "Can't merge {} into {}: file exists: " "{}".format(src, dst, existing[0])
        )

    locks = []
    if locking:
        if not lock:
            lock = get_lock(src, timeout=timeout)
        locks = [lock]
    with try_acquire_locks(locks, timeout):
        copytree(src, dst, symlinks=symlinks)


def get_prefix_replacement_paths(src, dst):
    ssplit = src.split(os.path.sep)
    dsplit = dst.split(os.path.sep)
    while ssplit and ssplit[-1] == dsplit[-1]:
        del ssplit[-1]
        del dsplit[-1]
    return os.path.join(*ssplit), os.path.join(*dsplit)


def copy_into(
    src, dst, timeout=900, symlinks=False, lock=None, locking=True, clobber=False
):
    """Copy all the files and directories in src to the directory dst"""

    if symlinks and islink(src):
        try:
            os.makedirs(os.path.dirname(dst))
        except OSError:
            pass
        if os.path.lexists(dst):
            os.remove(dst)
        src_base, dst_base = get_prefix_replacement_paths(src, dst)
        src_target = os.readlink(src)
        src_replaced = src_target.replace(src_base, dst_base)
        os.symlink(src_replaced, dst)
        try:
            st = os.lstat(src)
            mode = stat.S_IMODE(st.st_mode)
            os.lchmod(dst, mode)
        except:
            pass  # lchmod not available
    elif isdir(src):
        merge_tree(
            src,
            dst,
            symlinks,
            timeout=timeout,
            lock=lock,
            locking=locking,
            clobber=clobber,
        )

    else:
        if isdir(dst):
            dst_fn = os.path.join(dst, os.path.basename(src))
        else:
            dst_fn = dst

        if os.path.isabs(src):
            src_folder = os.path.dirname(src)
        else:
            if os.path.sep in dst_fn:
                src_folder = os.path.dirname(dst_fn)
                if not os.path.isdir(src_folder):
                    os.makedirs(src_folder)
            else:
                src_folder = os.getcwd()

        if os.path.islink(src) and not os.path.exists(os.path.realpath(src)):
            log.warn("path %s is a broken symlink - ignoring copy", src)
            return

        if not lock and locking:
            lock = get_lock(src_folder, timeout=timeout)
        locks = [lock] if locking else []
        with try_acquire_locks(locks, timeout):
            # if intermediate folders not not exist create them
            dst_folder = os.path.dirname(dst)
            if dst_folder and not os.path.exists(dst_folder):
                try:
                    os.makedirs(dst_folder)
                except OSError:
                    pass
            try:
                _copy_with_shell_fallback(src, dst_fn)
            except shutil.Error:
                log.debug(
                    "skipping %s - already exists in %s", os.path.basename(src), dst
                )


def move_with_fallback(src, dst):
    try:
        shutil.move(src, dst)
    except PermissionError:
        try:
            copy_into(src, dst)
            os.unlink(src)
        except PermissionError:

            log.debug(
                f"Failed to copy/remove path from {src} to {dst} due to permission error"
            )
