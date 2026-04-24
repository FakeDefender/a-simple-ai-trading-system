"""市场规则与执行约束抽象。"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


DEFAULT_MARKET_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "timezone": "Asia/Shanghai",
        "trading_days": [0, 1, 2, 3, 4],
        "sessions": [{"start": "09:30", "end": "15:00"}],
        "exit_only_start": "14:55",
        "allow_short": True,
        "allow_fractional": False,
        "lot_size": 1.0,
        "quantity_precision": 6,
    },
    "us_equity": {
        "timezone": "America/New_York",
        "trading_days": [0, 1, 2, 3, 4],
        "sessions": [{"start": "09:30", "end": "16:00"}],
        "exit_only_start": "15:55",
        "allow_short": True,
        "allow_fractional": False,
        "lot_size": 1.0,
        "quantity_precision": 6,
    },
    "cn_equity": {
        "timezone": "Asia/Shanghai",
        "trading_days": [0, 1, 2, 3, 4],
        "sessions": [
            {"start": "09:30", "end": "11:30"},
            {"start": "13:00", "end": "15:00"},
        ],
        "exit_only_start": "14:57",
        "allow_short": False,
        "allow_fractional": False,
        "lot_size": 100.0,
        "quantity_precision": 0,
    },
    "hk_equity": {
        "timezone": "Asia/Hong_Kong",
        "trading_days": [0, 1, 2, 3, 4],
        "sessions": [
            {"start": "09:30", "end": "12:00"},
            {"start": "13:00", "end": "16:00"},
        ],
        "exit_only_start": "15:55",
        "allow_short": False,
        "allow_fractional": False,
        "lot_size": 100.0,
        "quantity_precision": 0,
    },
    "crypto_spot": {
        "timezone": "UTC",
        "trading_days": [0, 1, 2, 3, 4, 5, 6],
        "sessions": [{"start": "00:00", "end": "23:59"}],
        "exit_only_start": None,
        "allow_short": False,
        "allow_fractional": True,
        "lot_size": 0.0001,
        "quantity_precision": 6,
    },
}

SESSION_SETTING_KEYS = {
    "timezone",
    "trading_days",
    "sessions",
    "session_start",
    "session_end",
    "exit_only_start",
}

CONSTRAINT_SETTING_KEYS = {
    "allow_short",
    "allow_fractional",
    "lot_size",
    "quantity_precision",
}

MARKET_SETTING_KEYS = SESSION_SETTING_KEYS | CONSTRAINT_SETTING_KEYS


@dataclass(frozen=True)
class MarketProfile:
    name: str
    timezone: str
    trading_days: Tuple[int, ...]
    sessions: Tuple[Tuple[str, str], ...]
    exit_only_start: Optional[str]
    allow_short: bool
    allow_fractional: bool
    lot_size: float
    quantity_precision: int

    def to_dict(self) -> Dict[str, Any]:
        sessions = [{"start": start, "end": end} for start, end in self.sessions]
        payload: Dict[str, Any] = {
            "market_profile": self.name,
            "timezone": self.timezone,
            "trading_days": list(self.trading_days),
            "sessions": sessions,
            "allow_short": bool(self.allow_short),
            "allow_fractional": bool(self.allow_fractional),
            "lot_size": float(self.lot_size),
            "quantity_precision": int(self.quantity_precision),
        }
        if sessions:
            payload["session_start"] = sessions[0]["start"]
            payload["session_end"] = sessions[-1]["end"]
        if self.exit_only_start:
            payload["exit_only_start"] = self.exit_only_start
        return payload


def _normalize_symbol(symbol: Optional[str]) -> str:
    return str(symbol or "").strip().lower()


def _parse_sessions(
    sessions: Optional[Iterable[Any]] = None,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
) -> Tuple[Tuple[str, str], ...]:
    normalized = []
    if sessions:
        for session in sessions:
            if isinstance(session, dict):
                start = session.get("start")
                end = session.get("end")
            else:
                start, end = str(session).split("-", 1)
            if not start or not end:
                continue
            normalized.append((str(start), str(end)))
    elif session_start and session_end:
        normalized.append((str(session_start), str(session_end)))

    if not normalized:
        default_profile = DEFAULT_MARKET_PROFILES["default"]
        normalized = [(item["start"], item["end"]) for item in default_profile["sessions"]]

    normalized = sorted(normalized, key=lambda item: item[0])
    return tuple(normalized)


def _infer_profile_name(symbol: Optional[str]) -> str:
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return "default"

    if normalized_symbol.endswith((".sh", ".sz", ".bj")):
        return "cn_equity"
    if normalized_symbol.endswith(".hk"):
        return "hk_equity"
    if normalized_symbol.endswith(".us"):
        return "us_equity"
    if any(token in normalized_symbol for token in ("/", "-")) and any(
        quote in normalized_symbol for quote in ("usd", "usdt", "btc", "eth")
    ):
        return "crypto_spot"
    return "default"


class MarketProfileResolver:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.market_config = dict(self.config.get("market", {}) or {})
        raw_symbol_profiles = dict(self.market_config.get("symbol_profiles", {}) or {})
        self.symbol_profiles = {str(symbol): value for symbol, value in raw_symbol_profiles.items()}
        self.normalized_symbol_profiles = {_normalize_symbol(symbol): value for symbol, value in self.symbol_profiles.items()}
        raw_symbol_overrides = dict(self.market_config.get("symbol_overrides", {}) or {})
        self.symbol_overrides = {str(symbol): dict(value or {}) for symbol, value in raw_symbol_overrides.items()}
        self.normalized_symbol_overrides = {
            _normalize_symbol(symbol): dict(value or {}) for symbol, value in self.symbol_overrides.items()
        }

    def _market_overrides(self) -> Dict[str, Any]:
        return {key: value for key, value in self.market_config.items() if key in MARKET_SETTING_KEYS}

    def _section_config(self, section_name: Optional[str]) -> Dict[str, Any]:
        if not section_name:
            return {}
        return dict(self.config.get(section_name, {}) or {})

    def _symbol_profile_config(self, symbol: Optional[str]) -> Dict[str, Any]:
        normalized_symbol = _normalize_symbol(symbol)
        if not normalized_symbol:
            return {}
        profile = self.normalized_symbol_profiles.get(normalized_symbol)
        if isinstance(profile, dict):
            return dict(profile)
        return {}

    def _symbol_override_config(self, symbol: Optional[str]) -> Dict[str, Any]:
        normalized_symbol = _normalize_symbol(symbol)
        if not normalized_symbol:
            return {}
        override = self.normalized_symbol_overrides.get(normalized_symbol)
        return dict(override or {})

    def _profile_name(self, symbol: Optional[str], section_name: Optional[str]) -> str:
        normalized_symbol = _normalize_symbol(symbol)
        section_config = self._section_config(section_name)

        symbol_profile = self.normalized_symbol_profiles.get(normalized_symbol)
        if isinstance(symbol_profile, dict):
            symbol_profile = symbol_profile.get("profile")

        explicit_profile = (
            symbol_profile
            or section_config.get("market_profile")
            or self.market_config.get("profile")
            or "auto"
        )
        explicit_profile = str(explicit_profile or "auto").strip().lower()
        if explicit_profile in {"", "auto", "infer"}:
            return _infer_profile_name(symbol)
        if explicit_profile in DEFAULT_MARKET_PROFILES:
            return explicit_profile
        return "default"

    def _has_symbol_specific_market(self, symbol: Optional[str]) -> bool:
        normalized_symbol = _normalize_symbol(symbol)
        return bool(
            normalized_symbol
            and (
                normalized_symbol in self.normalized_symbol_profiles
                or normalized_symbol in self.normalized_symbol_overrides
            )
        )

    def _resolve_session_payload(
        self,
        base_profile: Dict[str, Any],
        market_overrides: Dict[str, Any],
        symbol_profile_config: Dict[str, Any],
        symbol_override_config: Dict[str, Any],
        section_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        session_payload: Dict[str, Any] = {
            "sessions": base_profile.get("sessions"),
            "session_start": base_profile.get("session_start"),
            "session_end": base_profile.get("session_end"),
        }

        for source in [market_overrides, symbol_profile_config, symbol_override_config, section_config]:
            if not source:
                continue
            if "sessions" in source:
                session_payload["sessions"] = source.get("sessions")
                session_payload["session_start"] = source.get("session_start")
                session_payload["session_end"] = source.get("session_end")
            if "session_start" in source:
                session_payload["session_start"] = source.get("session_start")
                session_payload.pop("sessions", None)
            if "session_end" in source:
                session_payload["session_end"] = source.get("session_end")
                session_payload.pop("sessions", None)

        return session_payload

    def resolve(self, symbol: Optional[str] = None, section_name: Optional[str] = None) -> MarketProfile:
        profile_name = self._profile_name(symbol, section_name)
        base_profile = dict(DEFAULT_MARKET_PROFILES.get(profile_name, DEFAULT_MARKET_PROFILES["default"]))
        market_overrides = self._market_overrides()
        section_config = self._section_config(section_name)
        symbol_profile_config = self._symbol_profile_config(symbol)
        symbol_override_config = self._symbol_override_config(symbol)
        has_symbol_specific_market = self._has_symbol_specific_market(symbol)

        merged: Dict[str, Any] = {}
        merged.update(base_profile)
        merged.update(market_overrides)
        if not has_symbol_specific_market:
            merged.update({key: value for key, value in section_config.items() if key in CONSTRAINT_SETTING_KEYS})
        merged.update(symbol_profile_config)
        merged.update(symbol_override_config)
        merged.update({key: value for key, value in section_config.items() if key in {"timezone", "trading_days", "exit_only_start"}})

        session_payload = self._resolve_session_payload(
            base_profile=base_profile,
            market_overrides=market_overrides,
            symbol_profile_config=symbol_profile_config,
            symbol_override_config=symbol_override_config,
            section_config=section_config,
        )
        sessions = _parse_sessions(
            sessions=session_payload.get("sessions"),
            session_start=session_payload.get("session_start"),
            session_end=session_payload.get("session_end"),
        )

        strategy_allow_short = bool(self.config.get("strategy", {}).get("allow_short", True))
        allow_short = bool(merged.get("allow_short", True)) and strategy_allow_short
        allow_fractional = bool(merged.get("allow_fractional", False))
        lot_size = float(merged.get("lot_size", 1.0) or 1.0)
        quantity_precision = int(merged.get("quantity_precision", 6))

        return MarketProfile(
            name=profile_name,
            timezone=str(merged.get("timezone", DEFAULT_MARKET_PROFILES["default"]["timezone"])),
            trading_days=tuple(int(day) for day in merged.get("trading_days", DEFAULT_MARKET_PROFILES["default"]["trading_days"])),
            sessions=sessions,
            exit_only_start=merged.get("exit_only_start"),
            allow_short=allow_short,
            allow_fractional=allow_fractional,
            lot_size=lot_size,
            quantity_precision=quantity_precision,
        )

    def section_settings(self, symbol: Optional[str] = None, section_name: Optional[str] = None) -> Dict[str, Any]:
        return self.resolve(symbol=symbol, section_name=section_name).to_dict()
