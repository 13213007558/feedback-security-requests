"""Tests for the independent netrc admission policy.

The netrc admission policy is decoupled from ``Session.trust_env``: proxy and
CA bundle configuration keeps following the environment while automatic netrc
credentials are gated per request origin. All interactions here use a fake
adapter and a temporary netrc file, so no real network or user home directory
is touched.
"""

from __future__ import annotations

import base64
import io
import os
from unittest import mock

import pytest

import requests
from requests import (
    NETRC_AUTH_DISABLED,
    AllowedHostsNetrcAuth,
    DenyAllNetrcAuth,
    NetrcAuthPolicy,
    Session,
)
from requests._internal_utils import UNSET
from requests.compat import urlparse
from requests.sessions import SessionRedirectMixin

HOST_A = "http://host-a.example/"
HOST_B = "http://host-b.example/"
HTTPS_HOST_B = "https://host-b.example/"

_PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "ftp_proxy",
    "grpc_proxy",
    "rsync_proxy",
    "docker_http_proxy",
    "docker_https_proxy",
)


@pytest.fixture(autouse=True)
def clean_environment():
    """Pin proxy/CA env vars so the sandbox cannot influence proxy selection."""
    saved = dict(os.environ)
    for name in _PROXY_VARS:
        os.environ.pop(name, None)
        os.environ.pop(name.upper(), None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)
    os.environ.pop("CURL_CA_BUNDLE", None)
    os.environ.pop("NETRC", None)
    with mock.patch.dict(os.environ, {}, clear=False):
        yield
    os.environ.clear()
    os.environ.update(saved)


def _basic_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


class FakeRaw(io.BytesIO):
    """Minimal stand-in for a urllib3 response object."""

    def __init__(self) -> None:
        super().__init__(b"")
        # Without ``_original_response`` requests skips cookie extraction.
        self._original_response = None

    def release_conn(self) -> None:
        pass


class FakeAdapter:
    """Records each send call and can redirect to another host."""

    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        self.redirects = redirects or {}
        self.calls: list[tuple[requests.PreparedRequest, dict]] = []

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        response = requests.Response()
        response.status_code = 200
        response.headers = requests.structures.CaseInsensitiveDict()
        response.raw = FakeRaw()
        response.request = request
        response.url = request.url
        target = self.redirects.get(request.url)
        if target is not None:
            response.status_code = 302
            response.headers["Location"] = target
        return response

    def close(self) -> None:
        pass


class RecordingPolicy(NetrcAuthPolicy):
    """Policy that records every admission decision and lookup."""

    def __init__(self, allowed: set[str] | None = None) -> None:
        self.allowed = allowed
        self.decisions: list[str] = []

    def is_allowed(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        self.decisions.append(host)
        return self.allowed is None or host in self.allowed


def make_netrc(tmp_path, monkeypatch, entries: dict[str, tuple[str, str]]) -> str:
    """Create a temporary netrc file and point NETRC at it."""
    netrc_path = tmp_path / "netrc"
    lines = []
    for machine, (login, password) in entries.items():
        lines.append(f"machine {machine}\nlogin {login}\npassword {password}\n")
    netrc_path.write_text("".join(lines))
    monkeypatch.setenv("NETRC", str(netrc_path))
    return str(netrc_path)


@pytest.fixture
def session_with_fake_adapter():
    session = Session()
    adapter = FakeAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


class TestPolicyBasics:
    def test_default_policy_allows_every_origin(self):
        assert NetrcAuthPolicy()("http://anything.example/") is True

    def test_deny_all_policy_rejects_every_origin(self):
        assert DenyAllNetrcAuth()(HOST_A) is False
        assert NETRC_AUTH_DISABLED(HOST_B) is False

    def test_allowed_hosts_policy_matches_case_insensitively(self):
        policy = AllowedHostsNetrcAuth(["host-a.example"])
        assert policy(HOST_A) is True
        assert policy("http://HOST-A.example/") is True
        assert policy(HOST_B) is False

    def test_allowed_hosts_policy_rejects_url_without_hostname(self):
        policy = AllowedHostsNetrcAuth(["host-a.example"])
        assert policy.is_allowed("http://") is False

    def test_custom_policy_subclass_hook(self):
        class OnlyHttp(NetrcAuthPolicy):
            def is_allowed(self, url):
                return urlparse(url).scheme == "http"

        policy = OnlyHttp()
        assert policy(HOST_A) is True
        assert policy(HTTPS_HOST_B) is False


class TestFirstRequestAdmission:
    def test_allowed_origin_receives_netrc_credentials(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("alice", "secret")})
        session.netrc_auth_policy = AllowedHostsNetrcAuth(["host-a.example"])

        session.get(HOST_A)

        request, _ = adapter.calls[0]
        assert request.headers["Authorization"] == _basic_header("alice", "secret")

    def test_denied_origin_does_not_read_netrc(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-b.example": ("bob", "secret")})
        session.netrc_auth_policy = AllowedHostsNetrcAuth(["host-a.example"])
        lookup_calls = []

        def counting_lookup(url):
            lookup_calls.append(url)
            return requests.utils.get_netrc_auth(url)

        monkeypatch.setattr(requests.sessions, "get_netrc_auth", counting_lookup)

        session.get(HOST_B)

        request, _ = adapter.calls[0]
        assert "Authorization" not in request.headers
        assert lookup_calls == []

    def test_disabled_policy_never_looks_up_credentials(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = NETRC_AUTH_DISABLED
        lookup_calls = []
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            lambda url: lookup_calls.append(url),
        )

        session.get(HOST_A)

        assert "Authorization" not in adapter.calls[0][0].headers
        assert lookup_calls == []

    def test_lookup_count_proves_netrc_file_is_not_opened_for_denied_origin(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, _ = session_with_fake_adapter
        netrc_path = make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = AllowedHostsNetrcAuth(["host-a.example"])

        real_exists = os.path.exists
        opens = []

        def tracking_exists(path):
            if path == netrc_path:
                opens.append(path)
            return real_exists(path)

        monkeypatch.setattr(os.path, "exists", tracking_exists)

        session.get(HOST_B)
        assert opens == []

        session.get(HOST_A)
        assert opens == [netrc_path]


class TestExplicitAuthPrecedence:
    def test_explicit_auth_is_not_overridden_by_netrc(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("netrc", "netrc")})
        session.netrc_auth_policy = NetrcAuthPolicy()

        session.get(HOST_A, auth=("explicit", "pw"))

        request, _ = adapter.calls[0]
        assert request.headers["Authorization"] == _basic_header("explicit", "pw")

    def test_session_auth_is_not_overridden_by_netrc(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("netrc", "netrc")})
        session.auth = ("session", "pw")
        session.netrc_auth_policy = NetrcAuthPolicy()

        session.get(HOST_A)

        request, _ = adapter.calls[0]
        assert request.headers["Authorization"] == _basic_header("session", "pw")

    def test_explicitly_empty_auth_tuple_is_not_treated_as_missing(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("netrc", "netrc")})
        session.netrc_auth_policy = NetrcAuthPolicy()
        lookup_calls = []
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            lambda url: lookup_calls.append(url),
        )

        session.get(HOST_A, auth=())

        request, _ = adapter.calls[0]
        assert "Authorization" not in request.headers
        assert lookup_calls == []

    def test_falsy_session_auth_is_not_treated_as_missing(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("netrc", "netrc")})
        session.auth = ()
        session.netrc_auth_policy = NetrcAuthPolicy()
        lookup_calls = []
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            lambda url: lookup_calls.append(url),
        )

        session.get(HOST_A)

        assert "Authorization" not in adapter.calls[0][0].headers
        assert lookup_calls == []


class TestRedirectRebuild:
    def test_credentials_rebuilt_on_cross_host_redirect_when_allowed(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        adapter.redirects = {HOST_A: HOST_B}
        make_netrc(
            tmp_path,
            monkeypatch,
            {
                "host-a.example": ("alice", "pw-a"),
                "host-b.example": ("bob", "pw-b"),
            },
        )
        policy = RecordingPolicy()
        session.netrc_auth_policy = policy

        response = session.get(HOST_A)

        assert [urlparse(req.url).hostname for req, _ in adapter.calls] == [
            "host-a.example",
            "host-b.example",
        ]
        first, second = (req for req, _ in adapter.calls)
        assert first.headers["Authorization"] == _basic_header("alice", "pw-a")
        assert second.headers["Authorization"] == _basic_header("bob", "pw-b")
        assert policy.decisions == ["host-a.example", "host-b.example"]
        assert response.url == HOST_B

    def test_denied_redirect_target_skips_netrc_and_strips_old_credentials(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        adapter.redirects = {HOST_A: HOST_B}
        make_netrc(
            tmp_path,
            monkeypatch,
            {
                "host-a.example": ("alice", "pw-a"),
                "host-b.example": ("bob", "pw-b"),
            },
        )
        session.netrc_auth_policy = AllowedHostsNetrcAuth(["host-a.example"])
        lookup_urls = []

        def counting_lookup(url):
            lookup_urls.append(url)
            return requests.utils.get_netrc_auth(url)

        monkeypatch.setattr(requests.sessions, "get_netrc_auth", counting_lookup)

        session.get(HOST_A)

        first, second = (req for req, _ in adapter.calls)
        assert first.headers["Authorization"] == _basic_header("alice", "pw-a")
        assert "Authorization" not in second.headers
        assert lookup_urls == [HOST_A]

    def test_policy_is_evaluated_for_every_redirect_hop(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        adapter.redirects = {HOST_A: HOST_B, HOST_B: "http://host-c.example/"}
        make_netrc(
            tmp_path,
            monkeypatch,
            {
                "host-a.example": ("a", "p"),
                "host-b.example": ("b", "p"),
                "host-c.example": ("c", "p"),
            },
        )
        policy = RecordingPolicy(allowed={"host-a.example", "host-c.example"})
        session.netrc_auth_policy = policy

        session.get(HOST_A)

        requests_sent = [req for req, _ in adapter.calls]
        assert "Authorization" in requests_sent[0].headers
        assert "Authorization" not in requests_sent[1].headers
        assert "Authorization" in requests_sent[2].headers
        assert policy.decisions == [
            "host-a.example",
            "host-b.example",
            "host-c.example",
        ]


class TestEnvironmentIndependence:
    def test_trust_env_false_disables_netrc_without_policy(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.trust_env = False
        lookup_calls = []
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            lambda url: lookup_calls.append(url),
        )

        session.get(HOST_A)

        assert "Authorization" not in adapter.calls[0][0].headers
        assert lookup_calls == []

    def test_policy_enables_netrc_while_trust_env_false(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.trust_env = False
        session.netrc_auth_policy = AllowedHostsNetrcAuth(["host-a.example"])

        session.get(HOST_A)

        assert adapter.calls[0][0].headers["Authorization"] == _basic_header("a", "b")

    def test_proxy_and_ca_settings_survive_when_netrc_denied(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = NETRC_AUTH_DISABLED
        ca_bundle = str(tmp_path / "ca-bundle.pem")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_bundle)
        monkeypatch.setenv("http_proxy", "http://proxy.example:3128")

        session.get(HOST_A)

        request, kwargs = adapter.calls[0]
        assert "Authorization" not in request.headers
        assert kwargs["verify"] == ca_bundle
        assert kwargs["proxies"]["http"] == "http://proxy.example:3128"

    def test_proxy_rebuild_follows_no_proxy_across_redirect(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        adapter.redirects = {HOST_A: HOST_B}
        make_netrc(
            tmp_path,
            monkeypatch,
            {"host-a.example": ("a", "p"), "host-b.example": ("b", "p")},
        )
        session.netrc_auth_policy = NETRC_AUTH_DISABLED
        monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
        monkeypatch.setenv("no_proxy", "host-b.example")

        session.get(HOST_A)

        first_request, first_kwargs = adapter.calls[0]
        second_request, second_kwargs = adapter.calls[1]
        assert first_kwargs["proxies"]["http"] == "http://proxy.example:3128"
        # NO_PROXY covers the second host: requests resolves a bypass even
        # though the carried-over proxy mapping still lists the scheme key.
        assert (
            requests.utils.should_bypass_proxies(
                second_request.url, no_proxy=second_kwargs["proxies"].get("no")
            )
            is True
        )
        assert (
            requests.utils.should_bypass_proxies(
                first_request.url, no_proxy=first_kwargs["proxies"].get("no")
            )
            is False
        )
        for request, _ in adapter.calls:
            assert "Authorization" not in request.headers


class TestRequestLevelOverride:
    def test_request_disable_overrides_permissive_session_policy(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = NetrcAuthPolicy()
        lookup_calls = []
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            lambda url: lookup_calls.append(url),
        )

        session.get(HOST_A, netrc_auth_policy=False)

        assert "Authorization" not in adapter.calls[0][0].headers
        assert lookup_calls == []

    def test_request_policy_overrides_restrictive_session_policy(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = NETRC_AUTH_DISABLED

        session.get(
            HOST_A,
            netrc_auth_policy=AllowedHostsNetrcAuth(["host-a.example"]),
        )

        assert adapter.calls[0][0].headers["Authorization"] == _basic_header("a", "b")

    def test_request_true_uses_default_permissive_policy(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        session.netrc_auth_policy = NETRC_AUTH_DISABLED

        session.get(HOST_A, netrc_auth_policy=True)

        assert adapter.calls[0][0].headers["Authorization"] == _basic_header("a", "b")

    def test_override_applies_to_redirects_started_by_request(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        adapter.redirects = {HOST_A: HOST_B}
        make_netrc(
            tmp_path,
            monkeypatch,
            {"host-a.example": ("a", "p"), "host-b.example": ("b", "p")},
        )
        session.netrc_auth_policy = NetrcAuthPolicy()

        session.get(
            HOST_A,
            netrc_auth_policy=AllowedHostsNetrcAuth(["host-a.example"]),
        )

        first, second = (req for req, _ in adapter.calls)
        assert "Authorization" in first.headers
        assert "Authorization" not in second.headers


class TestLegacyCompatibility:
    def test_default_session_keeps_legacy_netrc_behavior(
        self, tmp_path, monkeypatch, session_with_fake_adapter
    ):
        session, adapter = session_with_fake_adapter
        make_netrc(tmp_path, monkeypatch, {"host-a.example": ("a", "b")})
        assert session.netrc_auth_policy is None

        session.get(HOST_A)

        assert adapter.calls[0][0].headers["Authorization"] == _basic_header("a", "b")

    def test_trust_env_false_usage_remains_supported(self):
        # Mirrors the documented pre-feature way of disabling netrc.
        session = Session()
        session.trust_env = False
        prepared = requests.Request("GET", HOST_A).prepare()
        session.rebuild_auth(prepared, _response_for(prepared))
        assert "Authorization" not in prepared.headers

    def test_redirect_mixin_without_explicit_policy_uses_trust_env(self):
        class LegacyMixinSession(SessionRedirectMixin):
            def __init__(self):
                self.trust_env = False
                self.max_redirects = 30
                self.cookies = {}

        mixin = LegacyMixinSession()
        prepared = requests.Request("GET", HOST_A).prepare()
        mixin.rebuild_auth(prepared, _response_for(prepared))
        assert prepared._netrc_auth_policy is NETRC_AUTH_DISABLED

    def test_session_pickle_round_trip_preserves_policy(self):
        import pickle

        session = Session()
        policy = AllowedHostsNetrcAuth(["host-a.example"])
        session.netrc_auth_policy = policy
        restored = pickle.loads(pickle.dumps(session))
        assert restored.netrc_auth_policy.allowed_hosts == policy.allowed_hosts

    def test_unset_sentinel_is_falsy_and_singular(self):
        assert bool(UNSET) is False
        assert UNSET is type(UNSET)()


def _response_for(prepared):
    response = requests.Response()
    response.status_code = 200
    response.headers = requests.structures.CaseInsensitiveDict()
    response.raw = FakeRaw()
    response.request = prepared
    response.url = prepared.url
    return response
