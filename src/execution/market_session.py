"""交易时段控制。"""

from typing import Any, Iterable, List, Optional, Set, Tuple

import pandas as pd


class MarketSession:
    def __init__(
        self,
        timezone: str = "Asia/Shanghai",
        trading_days: Optional[Iterable[int]] = None,
        session_start: str = "09:30",
        session_end: str = "15:00",
        exit_only_start: Optional[str] = None,
        sessions: Optional[Iterable[Any]] = None,
    ):
        self.timezone = timezone
        self.trading_days: Set[int] = set(int(day) for day in (trading_days or [0, 1, 2, 3, 4]))
        self.sessions_minutes: List[Tuple[int, int]] = self._parse_sessions(sessions, session_start, session_end)
        self.session_start_minutes = self.sessions_minutes[0][0]
        self.session_end_minutes = self.sessions_minutes[-1][1]
        self.exit_only_start_minutes = self._parse_time(exit_only_start) if exit_only_start else None

    def _parse_time(self, value: str) -> int:
        hour_text, minute_text = str(value).split(":", 1)
        return int(hour_text) * 60 + int(minute_text)

    def _parse_sessions(
        self,
        sessions: Optional[Iterable[Any]],
        session_start: str,
        session_end: str,
    ) -> List[Tuple[int, int]]:
        parsed: List[Tuple[int, int]] = []
        if sessions:
            for session in sessions:
                if isinstance(session, dict):
                    start = session.get("start")
                    end = session.get("end")
                else:
                    start, end = str(session).split("-", 1)
                if not start or not end:
                    continue
                parsed.append((self._parse_time(str(start)), self._parse_time(str(end))))
        else:
            parsed.append((self._parse_time(session_start), self._parse_time(session_end)))

        parsed.sort(key=lambda item: item[0])
        return parsed

    def _normalize_timestamp(self, timestamp: Any) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is not None and self.timezone:
            return ts.tz_convert(self.timezone)
        return ts

    def _time_in_minutes(self, timestamp: Any) -> int:
        ts = self._normalize_timestamp(timestamp)
        return ts.hour * 60 + ts.minute

    def is_trading_day(self, timestamp: Any) -> bool:
        ts = self._normalize_timestamp(timestamp)
        return ts.weekday() in self.trading_days

    def _is_inside_session(self, current_minutes: int) -> bool:
        for start_minutes, end_minutes in self.sessions_minutes:
            if start_minutes <= current_minutes <= end_minutes:
                return True
        return False

    def get_state(self, timestamp: Any) -> str:
        if not self.is_trading_day(timestamp):
            return "closed"

        current_minutes = self._time_in_minutes(timestamp)
        if not self._is_inside_session(current_minutes):
            return "closed"

        if self.exit_only_start_minutes is not None and current_minutes >= self.exit_only_start_minutes:
            return "exit_only"

        return "open"

    def is_open(self, timestamp: Any) -> bool:
        return self.get_state(timestamp) in {"open", "exit_only"}

    def can_open_new_position(self, timestamp: Any) -> bool:
        return self.get_state(timestamp) == "open"

    def can_reduce_position(self, timestamp: Any) -> bool:
        return self.get_state(timestamp) in {"open", "exit_only"}
