"""Tests for post-fork transport rebuilding.

A :class:`~requests.Session` is normally created in a parent process and then
inherited by forked workers. These tests verify that a forked child replaces
the inherited transport resources (sockets and locks) before its first
request, that the parent keeps using its own session untouched, and that
cleanup in either process never reaches into the other one.

Everything runs locally over pipes: no network access and no timing based
sleeps are used. On platforms without :func:`os.fork` the whole module is
skipped with an explicit reason.
"""

import os
import select
import threading
import warnings

import pytest
from urllib3.util.retry import Retry

from requests import Request, Session
from requests.adapters import BaseAdapter, HTTPAdapter
from requests.models import Response

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="os.fork() is not available on this platform",
)

PIPE_TIMEOUT = 5


class FakeSocket:
    """A stand-in for an inherited transport descriptor."""

    def __init__(self):
        self.closed_by = None

    def close(self):
        if self.closed_by is not None:
            raise AssertionError("inherited fake connection was closed twice")
        self.closed_by = os.getpid()


class FakePool:
    """Minimal pool holding an idle connection guarded by a real lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.idle_connections = [FakeSocket()]
        self.clear_calls = 0

    def clear(self):
        # Rebuilding must never clear inherited pools: doing so takes the
        # inherited lock and closes sockets the parent process still owns.
        self.clear_calls += 1
        raise AssertionError("inherited pool must not be cleared after fork")

    def close(self):
        for connection in self.idle_connections:
            connection.close()
        self.idle_connections = []


class FakeRaw:
    reason = "OK"
    headers = {}
    status = 200

    def read(self, *args, **kwargs):
        return b""

    def release_conn(self):
        pass


class FakeResponse(Response):
    def __init__(self, request):
        super().__init__()
        self.status_code = 200
        self.headers = {}
        self.raw = FakeRaw()
        self.url = request.url
        self.request = request
        self._content = b""
        self._content_consumed = True


class RecordingAdapter(BaseAdapter):
    """Adapter with an explicit, safe rebuild_for_fork implementation."""

    def __init__(self):
        super().__init__()
        self.pool = FakePool()
        self.closed = False
        self.sent = 0
        self.rebuilds = 0

    def rebuild_for_fork(self):
        self.rebuilds += 1
        # Abandon the inherited pool without acquiring its lock or closing it.
        self.pool = FakePool()
        self.closed = False

    def send(self, request, **kwargs):
        self.sent += 1
        # Sending must use the pool owned by the current process.
        with self.pool.lock:
            connection = self.pool.idle_connections[0]
            if connection.closed_by is not None:
                raise AssertionError("request used a closed connection")
        return FakeResponse(request)

    def close(self):
        self.closed = True
        self.pool.close()


class UnrebuildableAdapter(BaseAdapter):
    """Adapter that does not opt into the rebuild contract."""

    def send(self, request, **kwargs):
        raise AssertionError("send must not be reached")

    def close(self):
        pass


def _read_message(fd):
    """Read a single length-prefixed message written by a forked child."""
    readable, _, _ = select.select([fd], [], [], PIPE_TIMEOUT)
    if not readable:
        raise AssertionError("timed out waiting for message from forked child")
    header = b""
    while len(header) < 4:
        chunk = os.read(fd, 4 - len(header))
        if not chunk:
            raise AssertionError("forked child closed the pipe early")
        header += chunk
    length = int.from_bytes(header, "big")
    payload = b""
    while len(payload) < length:
        chunk = os.read(fd, length - len(payload))
        if not chunk:
            raise AssertionError("forked child closed the pipe early")
        payload += chunk
    return payload


def _run_in_fork(read_fd, write_fd, child_fn):
    """Fork and run ``child_fn`` in the child, returning its exit status."""
    with warnings.catch_warnings():
        # The lock test deliberately holds a lock in another thread across
        # fork(); the interpreter warns about that, which is expected here.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:  # pragma: no cover - executed in the child process
        os.close(read_fd)
        try:
            child_fn(write_fd)
            status = 0
        except Exception:
            import traceback

            message = traceback.format_exc().encode("utf-8")
            os.write(write_fd, len(message).to_bytes(4, "big") + message)
            status = 1
        finally:
            os.close(write_fd)
            os._exit(status)
    os.close(write_fd)
    _, wait_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(wait_status)


def _fork_result(child_fn):
    """Run ``child_fn`` in a fork and return the message it sends back."""
    read_fd, write_fd = os.pipe()
    exit_code = _run_in_fork(read_fd, write_fd, child_fn)
    message = _read_message(read_fd)
    os.close(read_fd)
    assert exit_code == 0, f"forked child failed:\n{message.decode('utf-8')}"
    return message


def test_http_adapter_rebuild_preserves_connection_and_retry_settings():
    retry = Retry(total=3, connect=2)
    adapter = HTTPAdapter(
        pool_connections=4,
        pool_maxsize=7,
        max_retries=retry,
        pool_block=True,
    )
    old_manager = adapter.poolmanager

    adapter.rebuild_for_fork()

    assert adapter.poolmanager is not old_manager
    assert adapter._pool_connections == 4
    assert adapter._pool_maxsize == 7
    assert adapter._pool_block is True
    assert adapter.max_retries.total == 3
    assert adapter.max_retries.connect == 2
    assert adapter.poolmanager.connection_pool_kw["block"] is True


def test_http_adapter_rebuild_preserves_proxy_settings():
    adapter = HTTPAdapter()
    proxy_url = "http://proxy.example:3128"
    old_proxy_manager = adapter.proxy_manager_for(proxy_url)
    old_manager = adapter.poolmanager

    adapter.rebuild_for_fork()

    assert adapter.poolmanager is not old_manager
    assert list(adapter.proxy_manager) == [proxy_url]
    new_proxy_manager = adapter.proxy_manager[proxy_url]
    assert new_proxy_manager is not old_proxy_manager


def test_base_adapter_rebuild_contract_is_explicit():
    adapter = UnrebuildableAdapter()

    with pytest.raises(NotImplementedError, match="rebuild_for_fork"):
        adapter.rebuild_for_fork()


def test_session_rebuilds_automatically_when_process_changes():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)

    # Simulate inheritance across a fork without rebuilding yet.
    session._owner_pid = -1
    assert adapter.rebuilds == 0

    prepared = Request("GET", "http://example.test/").prepare()
    session.send(prepared)

    assert adapter.rebuilds == 1
    assert session._owner_pid == os.getpid()

    # Subsequent requests in the same process must not rebuild again.
    session.send(prepared)
    assert adapter.rebuilds == 1
    assert adapter.sent == 2


def test_session_rebuild_error_identifies_unsafe_adapter():
    session = Session()
    session.mount("http://", UnrebuildableAdapter())
    session._owner_pid = -1

    with pytest.raises(NotImplementedError, match="rebuild_for_fork"):
        session.send(Request("GET", "http://example.test/").prepare())


def test_forked_child_replaces_idle_inherited_connection():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)

    inherited_pool = adapter.pool
    inherited_socket = inherited_pool.idle_connections[0]

    def child(write_fd):
        child_pid = os.getpid()
        session.rebuild_for_fork()
        assert adapter.pool is not inherited_pool
        # The inherited idle connection is abandoned, never touched.
        assert inherited_socket.closed_by is None
        assert inherited_pool.clear_calls == 0
        payload = f"{child_pid}:{id(adapter.pool)}".encode("ascii")
        os.write(write_fd, len(payload).to_bytes(4, "big") + payload)

    message = _fork_result(child)

    child_pid_text, child_pool_id = message.decode("ascii").split(":")
    assert int(child_pid_text) != os.getpid()
    assert int(child_pool_id) != id(inherited_pool)
    # The parent still owns its original pool and idle connection.
    assert adapter.pool is inherited_pool
    assert inherited_socket.closed_by is None
    assert inherited_pool.clear_calls == 0


def test_forked_child_rebuilds_without_taking_inherited_locked_lock():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)

    inherited_pool = adapter.pool
    # A parent thread holds the pool lock across the fork, so any attempt by
    # the child to acquire it would deadlock.
    release = threading.Event()
    acquired = threading.Event()

    def hold_lock():
        with inherited_pool.lock:
            acquired.set()
            release.wait(PIPE_TIMEOUT)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(PIPE_TIMEOUT)
    # Prove the lock really is owned by the holder thread before forking.
    assert inherited_pool.lock.acquire(blocking=False) is False

    def child(write_fd):
        session.rebuild_for_fork()
        # The inherited lock remains owned by the parent-side holder thread.
        if inherited_pool.lock.acquire(blocking=False):
            inherited_pool.lock.release()
            raise AssertionError("inherited lock was unexpectedly free in child")
        payload = str(adapter.rebuilds).encode("ascii")
        os.write(write_fd, len(payload).to_bytes(4, "big") + payload)

    message = _fork_result(child)
    release.set()
    holder.join(PIPE_TIMEOUT)

    assert message == b"1"
    assert not holder.is_alive()
    assert inherited_pool.clear_calls == 0


def test_forked_child_first_request_uses_fresh_transport():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)
    inherited_pool = adapter.pool

    def child(write_fd):
        prepared = Request("GET", "http://example.test/work").prepare()
        response = session.send(prepared)
        assert response.status_code == 200
        assert adapter.pool is not inherited_pool
        assert adapter.sent == 1
        assert inherited_pool.clear_calls == 0
        payload = str(id(adapter.pool)).encode("ascii")
        os.write(write_fd, len(payload).to_bytes(4, "big") + payload)

    message = _fork_result(child)

    assert int(message) != id(inherited_pool)
    # The parent never sent anything and keeps its original transport.
    assert adapter.pool is inherited_pool
    assert adapter.sent == 0


def test_parent_and_forked_child_close_their_own_transport_only():
    session = Session()
    adapter = RecordingAdapter()
    session.mount("http://", adapter)

    inherited_pool = adapter.pool
    inherited_socket = inherited_pool.idle_connections[0]

    def child(write_fd):
        session.rebuild_for_fork()
        child_pool = adapter.pool
        child_socket = child_pool.idle_connections[0]
        session.close()
        assert adapter.closed is True
        assert child_socket.closed_by == os.getpid()
        assert inherited_socket.closed_by is None
        assert inherited_pool.clear_calls == 0
        payload = b"closed"
        os.write(write_fd, len(payload).to_bytes(4, "big") + payload)

    message = _fork_result(child)

    assert message == b"closed"
    assert inherited_socket.closed_by is None
    assert inherited_pool.clear_calls == 0

    session.close()
    assert inherited_socket.closed_by == os.getpid()


def test_forked_child_with_unsafe_adapter_reports_clear_error():
    session = Session()
    session.mount("http://", UnrebuildableAdapter())

    def child(write_fd):
        try:
            session.rebuild_for_fork()
        except NotImplementedError as error:
            payload = str(error).encode("utf-8")
            os.write(write_fd, len(payload).to_bytes(4, "big") + payload)
        else:
            payload = b"expected NotImplementedError was not raised"
            os.write(write_fd, len(payload).to_bytes(4, "big") + payload)

    message = _fork_result(child)
    assert b"rebuild_for_fork" in message
