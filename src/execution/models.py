"""Paper trading 执行层的数据模型。"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    quantity: float
    submitted_at: Any
    order_type: str = "market"
    requested_price: Optional[float] = None
    signal: Optional[int] = None
    status: str = "pending"
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: Optional[str] = None
    updated_at: Any = None
    canceled_at: Any = None
    rejected_at: Any = None
    status_reason: Optional[str] = None
    attempt_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return float(max(float(self.quantity) - float(self.filled_quantity), 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    tax: float
    total_fees: float
    notional: float
    filled_at: Any
    broker_order_id: Optional[str] = None
    realized_pnl: float = 0.0
    gross_realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Any = None
    updated_at: Any = None

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "long"
        if self.quantity < 0:
            return "short"
        return "flat"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side
        return payload


@dataclass
class AccountSnapshot:
    timestamp: Any
    cash: float
    market_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    gross_exposure: float
    net_exposure: float
    active_positions: int
    fees_paid: float = 0.0
    open_orders: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
