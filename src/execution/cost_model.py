"""执行成本模型。"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExecutionCostBreakdown:
    market_price: float
    fill_price: float
    notional: float
    commission: float
    slippage: float
    tax: float
    total_fees: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "market_price": float(self.market_price),
            "fill_price": float(self.fill_price),
            "notional": float(self.notional),
            "commission": float(self.commission),
            "slippage": float(self.slippage),
            "tax": float(self.tax),
            "total_fees": float(self.total_fees),
        }


class ExecutionCostModel:
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        fixed_commission: float = 0.0,
        min_commission: float = 0.0,
        sell_tax_rate: float = 0.0,
    ):
        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)
        self.fixed_commission = float(fixed_commission)
        self.min_commission = float(min_commission)
        self.sell_tax_rate = float(sell_tax_rate)

    def estimate(self, side: str, market_price: float, quantity: float) -> ExecutionCostBreakdown:
        market_price = float(market_price)
        quantity = abs(float(quantity))
        if quantity <= 0 or market_price <= 0:
            return ExecutionCostBreakdown(
                market_price=market_price,
                fill_price=market_price,
                notional=0.0,
                commission=0.0,
                slippage=0.0,
                tax=0.0,
                total_fees=0.0,
            )

        fill_price = market_price * (1 + self.slippage_rate) if side == "buy" else market_price * (1 - self.slippage_rate)
        notional = float(fill_price * quantity)
        commission = float(max(notional * self.commission_rate + self.fixed_commission, self.min_commission))
        slippage = float(abs(fill_price - market_price) * quantity)
        tax = float(notional * self.sell_tax_rate) if side == "sell" else 0.0
        total_fees = float(commission + tax)
        return ExecutionCostBreakdown(
            market_price=market_price,
            fill_price=float(fill_price),
            notional=notional,
            commission=commission,
            slippage=slippage,
            tax=tax,
            total_fees=total_fees,
        )
