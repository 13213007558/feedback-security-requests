"""
requests._no_proxy
~~~~~~~~~~~~~~~~~~

Strict, opt-in matching for ``NO_PROXY`` / ``no_proxy``.

The legacy matcher in :mod:`requests.utils` (which delegates to the
interpreter's proxy helpers) is kept unchanged for backwards compatibility.
The parser and matcher defined here only become active when a caller opts in,
so existing configurations keep their historical routing until they are
deliberately migrated.

The strict rules are deliberately explicit:

* a bare entry (``example.test``) matches that host only, never a host that
  merely shares a suffix (``evilexample.test``);
* a leading dot (``.example.test``) matches the domain and any sub-domain;
* an optional port pins the rule to one port, while a rule without a port
  only matches the default port of the URL scheme (80 for ``http``, 443 for
  ``https``);
* IP literals (including bracketed IPv6) and CIDR ranges only match literal
  IPs in the URL -- no DNS lookup is ever performed;
* malformed entries raise :class:`~requests.exceptions.InvalidProxyConfig`
  that identifies the offending entry.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Final

from .compat import urlparse
from .exceptions import InvalidProxyConfig

#: Scheme default ports used when a rule does not pin a port.
DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}

#: Entry that bypasses the proxy for every host.
WILDCARD: Final = "*"


class InvalidNoProxyRule(InvalidProxyConfig):
    """A single ``NO_PROXY`` entry could not be parsed.

    The :attr:`index` and :attr:`rule` attributes locate the offending entry
    inside the comma separated configuration so the source of the
    misconfiguration can be fixed instead of being silently ignored.
    """

    def __init__(self, rule: str, index: int, reason: str) -> None:
        self.rule = rule
        self.index = index
        self.reason = reason
        super().__init__(f"Invalid NO_PROXY entry #{index} {rule!r}: {reason}")


@dataclass(frozen=True)
class NoProxyRule:
    """One parsed ``NO_PROXY`` entry."""

    raw: str
    index: int
    subdomains: bool = False
    port: int | None = None
    network: ipaddress.IPv4Network | ipaddress.IPv6Network | None = None
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    host: str | None = None
    wildcard: bool = False

    def describe(self) -> str:
        """Return the canonical text of the rule for diagnostic reports."""
        if self.wildcard:
            return WILDCARD
        if self.network is not None:
            text = str(self.network)
            if self.network.version == 6:
                address, suffix = text.split("/", 1)
                text = f"[{address}]/{suffix}"
        elif self.ip is not None:
            text = f"[{self.ip}]"
        else:
            text = ("." if self.subdomains else "") + (self.host or "")
        if self.port is not None:
            text = f"{text}:{self.port}"
        return text


@dataclass(frozen=True)
class NoProxyMatch:
    """The rule that caused a URL to bypass the proxy, if any."""

    rule: NoProxyRule

    @property
    def matched(self) -> str:
        """Canonical text of the matched rule."""
        return self.rule.describe()


def _parse_port(port_text: str, rule: str, index: int) -> int:
    if not port_text or not port_text.isascii() or not port_text.isdigit():
        raise InvalidNoProxyRule(rule, index, "port must be a number")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise InvalidNoProxyRule(rule, index, "port must be between 1 and 65535")
    return port


def _split_host_port(rule: str, index: int) -> tuple[str, int | None]:
    """Split a rule into a host part and optional port.

    Bracketed IPv6 (``[::1]`` / ``[::1]:8080``) is handled explicitly so the
    colons inside the address are not mistaken for a port separator.
    """
    if rule.startswith("["):
        end = rule.find("]")
        if end == -1:
            raise InvalidNoProxyRule(
                rule, index, "missing closing bracket for IPv6 literal"
            )
        host_part = rule[: end + 1]
        remainder = rule[end + 1 :]
        if remainder == "":
            return host_part, None
        if not remainder.startswith(":"):
            raise InvalidNoProxyRule(
                rule, index, "unexpected text after the IPv6 literal"
            )
        return host_part, _parse_port(remainder[1:], rule, index)

    if ":" not in rule:
        return rule, None
    host_part, _, port_part = rule.rpartition(":")
    # A single colon with no host (":8080") is invalid.
    if not host_part:
        raise InvalidNoProxyRule(rule, index, "missing host")
    # An unbracketed value containing several colons is an IPv6 literal that
    # is missing its square brackets.
    if _looks_like_unbracketed_ipv6(rule):
        raise InvalidNoProxyRule(
            rule, index, "IPv6 literals must be wrapped in square brackets"
        )
    return host_part, _parse_port(port_part, rule, index)


def _parse_rule(entry: str, index: int) -> NoProxyRule:
    rule = entry.strip()
    if rule == WILDCARD:
        return NoProxyRule(raw=rule, index=index, wildcard=True)

    host_part, port = _split_host_port(rule, index)

    if not host_part:
        raise InvalidNoProxyRule(rule, index, "missing host")

    subdomains = host_part.startswith(".")
    if subdomains:
        host_part = host_part[1:]
    if not host_part:
        raise InvalidNoProxyRule(rule, index, "missing host after leading dot")
    if "." in host_part and any(not label for label in host_part.split(".")):
        raise InvalidNoProxyRule(rule, index, "empty label in host")

    if "/" in host_part:
        network_text = host_part
        if network_text.startswith("[") and network_text.endswith("]"):
            network_text = network_text[1:-1]
        try:
            network = ipaddress.ip_network(network_text, strict=False)
        except ValueError as exc:
            raise InvalidNoProxyRule(rule, index, str(exc)) from exc
        if port is not None:
            raise InvalidNoProxyRule(
                rule, index, "CIDR ranges cannot be combined with a port"
            )
        return NoProxyRule(raw=rule, index=index, port=port, network=network)

    bracketed = host_part.startswith("[") and host_part.endswith("]")
    ip_candidate = host_part[1:-1] if bracketed else host_part
    try:
        ip = ipaddress.ip_address(ip_candidate)
    except ValueError:
        ip = None
    if ip is not None:
        if subdomains:
            raise InvalidNoProxyRule(
                rule, index, "IP literals cannot use a leading dot"
            )
        return NoProxyRule(raw=rule, index=index, port=port, ip=ip)

    if bracketed:
        raise InvalidNoProxyRule(
            rule, index, "brackets are only valid around IPv6 literals"
        )
    if _looks_like_unbracketed_ipv6(host_part):
        raise InvalidNoProxyRule(
            rule, index, "IPv6 literals must be wrapped in square brackets"
        )
    if _looks_like_numeric_ipv4(host_part):
        raise InvalidNoProxyRule(
            rule, index, f"{host_part!r} is not a valid IPv4 address"
        )
    host = host_part.lower()
    return NoProxyRule(
        raw=rule, index=index, subdomains=subdomains, port=port, host=host
    )


def _looks_like_unbracketed_ipv6(value: str) -> bool:
    # IPv6 contains at least two colons; host:port only contains one.
    return value.count(":") >= 2


def _looks_like_numeric_ipv4(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(part.isascii() and part.isdigit() for part in parts)


def parse_no_proxy(no_proxy: str | None) -> list[NoProxyRule]:
    """Parse a comma separated ``NO_PROXY`` value into strict rules.

    Empty entries (including stray commas) are ignored. Malformed entries
    raise :class:`InvalidNoProxyRule` carrying the entry and its position.
    """
    if not no_proxy:
        return []
    rules: list[NoProxyRule] = []
    for index, entry in enumerate(no_proxy.split(",")):
        if not entry.strip():
            continue
        rules.append(_parse_rule(entry, index))
    return rules


def effective_port(parsed, url_port: int | None) -> int | None:
    """Return the port to compare rules against.

    An absent URL port resolves to the scheme's default port for known
    schemes; unknown schemes keep ``None`` so port-less rules cannot match
    them by accident.
    """
    if url_port is not None:
        return url_port
    return DEFAULT_PORTS.get(parsed.scheme)


def _port_matches(rule: NoProxyRule, port: int | None) -> bool:
    if rule.port is None:
        # A rule without a port only covers the scheme's default port.
        return port in DEFAULT_PORTS.values()
    return port == rule.port


def strict_no_proxy_match(url: str, rules: list[NoProxyRule]) -> NoProxyMatch | None:
    """Return the first strict rule matching ``url``, or ``None``.

    Only literal IPs in the URL can match IP or CIDR rules and no DNS
    resolution is attempted.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return None

    port = effective_port(parsed, parsed.port)
    lowered = hostname.lower()

    try:
        literal_ip = ipaddress.ip_address(lowered)
    except ValueError:
        literal_ip = None

    for rule in rules:
        if rule.wildcard:
            return NoProxyMatch(rule)

        if rule.network is not None:
            if literal_ip is not None and isinstance(
                literal_ip, type(rule.network.network_address)
            ):
                if literal_ip in rule.network:
                    return NoProxyMatch(rule)
            continue

        if rule.ip is not None:
            if literal_ip is not None and literal_ip == rule.ip:
                if _port_matches(rule, port):
                    return NoProxyMatch(rule)
            continue

        if literal_ip is not None:
            # A domain rule must never match an IP literal.
            continue

        if rule.host is None or not _port_matches(rule, port):
            continue

        if rule.subdomains:
            if lowered == rule.host or lowered.endswith("." + rule.host):
                return NoProxyMatch(rule)
        elif lowered == rule.host:
            return NoProxyMatch(rule)

    return None


def should_bypass_proxies_strict(url: str, no_proxy: str | None) -> NoProxyMatch | None:
    """Parse ``no_proxy`` and return the strict rule matching ``url``."""
    rules = parse_no_proxy(no_proxy)
    return strict_no_proxy_match(url, rules)


def env_no_proxy() -> str | None:
    """Read ``no_proxy`` preferring the lowercase variable, like curl/wget."""
    return os.environ.get("no_proxy") or os.environ.get("NO_PROXY")
