"""Tests for the opt-in strict ``NO_PROXY`` matcher and the proxy selection
comparison entry point.

Everything here runs offline: routing is observed through a fake adapter, so
no socket connection or DNS lookup is performed.
"""

import pytest

import requests
from requests import _no_proxy as np
from requests.adapters import BaseAdapter
from requests.exceptions import InvalidProxyConfig
from requests.models import Response
from requests.structures import CaseInsensitiveDict
from requests.utils import (
    ProxyComparison,
    compare_proxy_selection,
    sanitize_proxy_url,
    should_bypass_proxies,
)


class RecordingAdapter(BaseAdapter):
    """Adapter that records the resolved proxy dict instead of connecting."""

    def __init__(self):
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append(kwargs.get("proxies"))
        response = Response()
        response.status_code = 200
        response.headers = CaseInsensitiveDict()
        response._content = b""
        response.request = request
        response.url = request.url
        return response

    def close(self):
        pass


class TestStrictRuleParsing:
    @pytest.mark.parametrize(
        "value",
        (
            "example..test",
            ".",
            ":80",
            "example.test:0",
            "example.test:99999",
            "example.test:notaport",
            "10.0.0.0/40",
            "999.1.1.1",
            ".1.2.3.4",
            "[not-an-ip]",
            "[::1",
            "2001:db8::1",
            "10.0.0.0/24:80",
            "[::1]extra",
        ),
    )
    def test_invalid_rules_are_located(self, value):
        with pytest.raises(InvalidProxyConfig) as excinfo:
            np.parse_no_proxy(value)
        error = excinfo.value
        assert isinstance(error, np.InvalidNoProxyRule)
        assert error.index == 0
        assert error.rule == value

    def test_invalid_rule_reports_position_in_list(self):
        with pytest.raises(np.InvalidNoProxyRule) as excinfo:
            np.parse_no_proxy("good.test, 999.1.1.1")
        assert excinfo.value.index == 1
        assert excinfo.value.rule == "999.1.1.1"

    @pytest.mark.parametrize(
        "value",
        ("", "   ", "a.test,,b.test", ", ,"),
    )
    def test_blank_entries_are_ignored(self, value):
        assert all(rule.host for rule in np.parse_no_proxy(value))

    @pytest.mark.parametrize(
        "value,canonical",
        (
            ("example.test", "example.test"),
            (".example.test", ".example.test"),
            ("example.test:8080", "example.test:8080"),
            ("10.0.0.0/24", "10.0.0.0/24"),
            ("[2001:db8::/32]", "[2001:db8::]/32"),
            ("[::1]:8080", "[::1]:8080"),
            ("*", "*"),
        ),
    )
    def test_canonical_rule_text(self, value, canonical):
        (rule,) = np.parse_no_proxy(value)
        assert rule.describe() == canonical


class TestStrictMatching:
    @pytest.mark.parametrize(
        "no_proxy,url,expected",
        (
            # Exact host only: subdomains and pseudo suffixes stay proxied.
            ("example.test", "http://example.test/", True),
            ("example.test", "http://EXAMPLE.TEST/", True),
            ("example.test", "http://www.example.test/", False),
            ("example.test", "http://evilexample.test/", False),
            # A leading dot covers the domain and its subdomains.
            (".example.test", "http://example.test/", True),
            (".example.test", "http://a.b.example.test/", True),
            (".example.test", "http://evilexample.test/", False),
            # Default ports: rule without a port matches 80/443 only.
            ("example.test", "http://example.test:80/", True),
            ("example.test", "http://example.test:8080/", False),
            ("example.test", "https://example.test:443/", True),
            ("example.test", "https://example.test:8443/", False),
            # Explicit port pinning.
            ("example.test:8080", "http://example.test:8080/", True),
            ("example.test:8080", "http://example.test/", False),
            ("example.test:8080", "http://example.test:80/", False),
            # IPv4 CIDR only matches literal IPs, any port.
            ("10.0.0.0/24", "http://10.0.0.55/", True),
            ("10.0.0.0/24", "http://10.0.0.55:9999/", True),
            ("10.0.0.0/24", "http://10.0.1.1/", False),
            # Bracketed IPv6 literals and networks.
            ("[::1]", "http://[::1]/", True),
            ("[::1]", "http://[::1]:8080/", False),
            ("[::1]:8080", "http://[::1]:8080/", True),
            ("[2001:db8::/32]", "http://[2001:db8:9::1]/", True),
            ("[2001:db8::/32]", "http://[2002:db8::1]/", False),
            # Wildcard bypasses everything.
            ("*", "http://anything.example/", True),
        ),
    )
    def test_match_semantics(self, no_proxy, url, expected):
        match = np.should_bypass_proxies_strict(url, no_proxy)
        assert bool(match) is expected

    def test_ip_family_is_not_cross_matched(self):
        assert (
            np.should_bypass_proxies_strict("http://10.0.0.1/", "[2001:db8::/32]")
            is None
        )
        assert (
            np.should_bypass_proxies_strict("http://[::ffff:10.0.0.1]/", "10.0.0.0/24")
            is None
        )

    def test_domain_rule_does_not_match_ip_literal(self):
        assert (
            np.should_bypass_proxies_strict("http://10.0.0.1/", "10.0.0.1") is not None
        )
        assert np.should_bypass_proxies_strict("http://10.0.0.1/", "host.test") is None

    def test_should_bypass_strict_flag(self, monkeypatch):
        monkeypatch.setenv("no_proxy", "example.test")
        # Strict mode does not consult the OS proxy_bypass fallback.
        assert (
            should_bypass_proxies("http://www.example.test/", None, strict=True)
            is False
        )

    def test_invalid_config_raises_through_should_bypass(self):
        with pytest.raises(InvalidProxyConfig):
            should_bypass_proxies("http://example.test/", "999.1.1.1", strict=True)

    def test_no_dns_lookup_for_cidr(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("strict matching must not resolve DNS")

        monkeypatch.setattr("socket.getaddrinfo", fail)
        monkeypatch.setattr("socket.gethostbyname", fail)
        # A host name never reaches CIDR matching; only the literal IP matches.
        assert (
            np.should_bypass_proxies_strict("http://named.example/", "10.0.0.0/24")
            is None
        )
        assert (
            np.should_bypass_proxies_strict("http://10.0.0.7/", "10.0.0.0/24")
            is not None
        )


class TestSanitization:
    @pytest.mark.parametrize(
        "raw,expected",
        (
            (None, None),
            ("http://proxy.example:3128", "http://proxy.example:3128"),
            (
                "http://user:secret@proxy.example:3128",
                "http://user:***@proxy.example:3128",
            ),
            (
                "socks5://user:p%40ss@proxy.example:1080",
                "socks5://user:***@proxy.example:1080",
            ),
        ),
    )
    def test_password_is_redacted(self, raw, expected):
        assert sanitize_proxy_url(raw) == expected


class TestCompareProxySelection:
    @pytest.fixture
    def env(self, monkeypatch):
        monkeypatch.delenv("REQUESTS_STRICT_NO_PROXY", raising=False)
        monkeypatch.setenv("http_proxy", "http://env-proxy:3128")
        monkeypatch.setenv("no_proxy", "example.test")

    def test_reports_legacy_vs_strict_choice(self, env):
        report = compare_proxy_selection("http://www.example.test/")
        assert isinstance(report, ProxyComparison)
        # Legacy lenient behavior bypasses; strict mode keeps the proxy.
        assert report.legacy.bypassed is True
        assert report.legacy.proxy is None
        assert report.strict.bypassed is False
        assert report.strict.proxy == "http://env-proxy:3128"
        assert report.differs is True

    def test_exact_host_matches_in_both_modes(self, env):
        report = compare_proxy_selection("http://example.test/")
        assert report.legacy.bypassed is True
        assert report.strict.bypassed is True
        assert report.strict.matched_rule == "example.test"
        assert report.differs is False

    def test_pseudo_suffix_is_proxied_in_both_modes(self, env):
        report = compare_proxy_selection("http://evilexample.test/")
        assert report.legacy.proxy == "http://env-proxy:3128"
        assert report.strict.proxy == "http://env-proxy:3128"
        assert report.differs is False

    def test_report_hides_proxy_password(self, monkeypatch):
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.setenv("http_proxy", "http://user:secret@env-proxy:3128")
        report = compare_proxy_selection("http://service.example/")
        assert "secret" not in report.legacy.proxy
        assert "***" in report.legacy.proxy
        assert "secret" not in report.strict.proxy

    def test_strict_invalid_config_is_raised(self, env):
        with pytest.raises(InvalidProxyConfig) as excinfo:
            compare_proxy_selection("http://example.test/", no_proxy="999.1.1.1")
        assert excinfo.value.rule == "999.1.1.1"

    def test_session_helper_uses_session_proxies(self):
        session = requests.Session()
        session.trust_env = False
        session.proxies = {"http": "http://session-proxy:3128"}
        report = session.compare_proxy_selection(
            "http://www.example.test/", no_proxy="example.test"
        )
        assert report.legacy.bypassed is True
        assert report.strict.proxy == "http://session-proxy:3128"

    def test_hostless_url_goes_direct_without_a_named_rule(self):
        report = compare_proxy_selection("file:///etc/hosts", no_proxy="example.test")
        assert report.strict.bypassed is True
        assert report.strict.matched_rule is None
        assert report.strict.proxy is None
        assert report.differs is False


class TestSessionRouting:
    def _session(self, monkeypatch, strict):
        monkeypatch.delenv("REQUESTS_STRICT_NO_PROXY", raising=False)
        monkeypatch.setenv("http_proxy", "http://env-proxy:3128")
        monkeypatch.setenv("no_proxy", "example.test")
        session = requests.Session()
        session.strict_no_proxy = strict
        adapter = RecordingAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session, adapter

    @pytest.mark.parametrize("strict", (False, True))
    def test_adapter_observes_final_route(self, monkeypatch, strict):
        session, adapter = self._session(monkeypatch, strict)
        session.get("http://www.example.test/")
        (proxies,) = adapter.calls
        if strict:
            assert proxies.get("http") == "http://env-proxy:3128"
        else:
            assert "http" not in proxies

    def test_request_keyword_overrides_session_flag(self, monkeypatch):
        session, adapter = self._session(monkeypatch, strict=False)
        session.get("http://www.example.test/", strict_no_proxy=True)
        (proxies,) = adapter.calls
        assert proxies.get("http") == "http://env-proxy:3128"

    def test_environment_switch_enables_strict(self, monkeypatch):
        monkeypatch.setenv("REQUESTS_STRICT_NO_PROXY", "true")
        monkeypatch.setenv("http_proxy", "http://env-proxy:3128")
        monkeypatch.setenv("no_proxy", "example.test")
        session = requests.Session()
        adapter = RecordingAdapter()
        session.mount("http://", adapter)
        session.get("http://www.example.test/")
        (proxies,) = adapter.calls
        assert proxies.get("http") == "http://env-proxy:3128"

    def test_cross_origin_redirect_reselects_proxy(self, monkeypatch):
        session, adapter = self._session(monkeypatch, strict=True)

        def routing_adapter(request, **kwargs):
            adapter.calls.append(kwargs.get("proxies"))
            response = Response()
            response.request = request
            response.url = request.url
            response._content = b""
            if request.url == "http://www.example.test/start":
                response.status_code = 302
                response.headers = CaseInsensitiveDict(
                    {"Location": "http://example.test/landing"}
                )
            else:
                response.status_code = 200
                response.headers = CaseInsensitiveDict()
            return response

        adapter.send = routing_adapter
        session.get("http://www.example.test/start")

        # First hop is proxied; the redirect target is an exact NO_PROXY host
        # and therefore goes direct.
        first, second = adapter.calls
        assert first.get("http") == "http://env-proxy:3128"
        assert "http" not in second

    def test_strict_bypass_keeps_host_specific_proxy(self, monkeypatch):
        monkeypatch.delenv("REQUESTS_STRICT_NO_PROXY", raising=False)
        session = requests.Session()
        session.strict_no_proxy = True
        adapter = RecordingAdapter()
        session.mount("http://", adapter)
        session.proxies = {
            "http": "http://generic:3128",
            "http://example.test": "http://dedicated:3128",
        }
        session.get(
            "http://example.test/",
            proxies={"no_proxy": "example.test"},
        )
        (proxies,) = adapter.calls
        assert "http" not in proxies
        assert proxies["http://example.test"] == "http://dedicated:3128"
