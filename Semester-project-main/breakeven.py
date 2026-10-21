"""Break-even model: This calculates the order volume at which it is feasible to ship directly 
from South Africa (SA) vs building a regional warehouse

We assume warehousing costs are in USD or Pounds and shipping costs are in ZAR

We consider each unit to be a game 

Direct shpping:
    Unit cost (ZAR)
    Total cost (ZAR)

Warehouse costs:
    Variable costs (US/UK)
    Fixed costs (US/UK)
    Holding costs (US/UK)
    Total regional costs =  Variable costs (US/UK) + Fixed costs (US/UK) + Holding costs (US/UK)


Break even : Total costs (direct) = Total cost (Warehouse)

Assumptions: Constant units costs (no frieght tiers). 
Steady demand woth normally distributed variation (CV.
One warehouse per region.
Each inbound shipment is the EOQ sized order.
Fixed costs remain constant for our time period.

"""

from dataclasses import dataclass, field, fields, replace
from math import sqrt
from typing import Dict, List, Optional


class ModelError(ValueError):
    """Raised when inputs are invalid."""


def _f(label, unit, group, default):
    return field(default=default, metadata={"label": label, "unit": unit, "group": group})


G_D, G_V, G_I, G_F = ("Direct shipping from SA", "Warehouse: per-unit costs",
                      "Inventory and replenishment", "Warehouse: fixed costs")


@dataclass(frozen=True)
class RegionInputs:
    fx_zar_per_usd: float = _f("Exchange rate", "ZAR per USD", G_D, 18.0)
    direct_freight_usd: float = _f("Freight per unit", "USD", G_D, 14.00)
    direct_customs_ins_usd: float = _f("Customs, duties and insurance per unit", "USD", G_D, 2.50)
    sa_handling_zar: float = _f("SA pick, pack and handling per unit", "ZAR", G_D, 20.0)
    inbound_freight_usd: float = _f("Bulk inbound freight per unit", "USD", G_V, 2.80)
    inbound_customs_ins_usd: float = _f("Inbound customs, duties and insurance per unit", "USD", G_V, 2.50)
    pick_pack_usd: float = _f("Pick and pack per unit", "USD", G_V, 1.80)
    local_delivery_usd: float = _f("Local delivery per unit", "USD", G_V, 4.50)
    unit_cost_zar: float = _f("Product cost per unit (values inventory)", "ZAR", G_I, 144.0)
    holding_rate_annual: float = _f("Annual holding rate (fraction of value)", "0-1", G_I, 0.20)
    lead_time_months: float = _f("Replenishment lead time (L)", "months", G_I, 1.5)
    demand_cv: float = _f("Demand variability (CV)", "ratio", G_I, 0.30)
    service_z: float = _f("Service level factor (z)", "z-score", G_I, 1.65)
    rent_usd_month: float = _f("Warehouse lease", "USD/month", G_F, 2500.0)
    staff_usd_month: float = _f("Warehouse staff", "USD/month", G_F, 3000.0)
    admin_usd_month: float = _f("Software, insurance and admin", "USD/month", G_F, 400.0)
    setup_usd: float = _f("One-off setup cost", "USD", G_F, 15000.0)
    amortisation_months: float = _f("Setup amortisation period", "months", G_F, 36.0)
    inbound_shipment_fixed_usd: float = _f("Fixed cost per inbound shipment (S)", "USD", G_I, 600.0)

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or v < 0:
                raise ModelError(f"{f.metadata['label']} must be a number of 0 or more.")
        for name in ("fx_zar_per_usd", "amortisation_months"):
            if getattr(self, name) <= 0:
                raise ModelError(f"{self.__dataclass_fields__[name].metadata['label']} must be above 0.")


def unit_value_usd(p: RegionInputs) -> float:
    return p.unit_cost_zar / p.fx_zar_per_usd


def direct_variable_cost(p: RegionInputs) -> float:
    """v_d (USD/unit)."""
    return p.direct_freight_usd + p.direct_customs_ins_usd + p.sa_handling_zar / p.fx_zar_per_usd


def annual_holding_cost_per_unit(p: RegionInputs) -> float:
    """H (USD per unit per year)."""
    return p.holding_rate_annual * unit_value_usd(p)


def safety_holding_cost_per_unit(p: RegionInputs) -> float:
    """Monthly carrying cost of safety stock per unit of monthly volume."""
    return annual_holding_cost_per_unit(p) / 12 * p.service_z * p.demand_cv * sqrt(p.lead_time_months)


def cycle_cost_coefficient(p: RegionInputs) -> float:
    """k in (monthly EOQ ordering + cycle-stock cost) = k * sqrt(Q)."""
    return sqrt(p.inbound_shipment_fixed_usd * annual_holding_cost_per_unit(p) / 6)


def warehouse_variable_cost(p: RegionInputs) -> float:
    """v_w (USD/unit)."""
    return (p.inbound_freight_usd + p.inbound_customs_ins_usd + p.pick_pack_usd
            + p.local_delivery_usd + safety_holding_cost_per_unit(p))


def warehouse_fixed_cost(p: RegionInputs) -> float:
    """F_w (USD/month)."""
    return (p.rent_usd_month + p.staff_usd_month + p.admin_usd_month
            + p.setup_usd / p.amortisation_months)


def calculate_eoq(demand: float, order_cost: float, hold_cost: float) -> float:
    """EOQ = sqrt(2*D*S/H). demand per year, order_cost per order, hold_cost per unit per year."""
    if hold_cost <= 0:
        raise ModelError("Holding cost must be above 0 to calculate an EOQ.")
    if demand < 0 or order_cost < 0:
        raise ModelError("Demand and ordering cost must be 0 or more.")
    return sqrt(2 * demand * order_cost / hold_cost)


def orders_per_year(demand: float, order_qty: float) -> float:
    return 0.0 if order_qty == 0 else demand / order_qty


def total_annual_cost(demand: float, order_cost: float, hold_cost: float, order_qty: float) -> float:
    """(D/Q)*S + (Q/2)*H."""
    return orders_per_year(demand, order_qty) * order_cost + order_qty / 2 * hold_cost


def reorder_point(p: RegionInputs, q: float) -> float:
    """Demand during lead time + safety stock (units)."""
    return q * p.lead_time_months + safety_stock(p, q)


def safety_stock(p: RegionInputs, q: float) -> float:
    return p.service_z * p.demand_cv * q * sqrt(p.lead_time_months)


def replenishment_plan(p: RegionInputs, q: float):
    """Warehouse restocking plan at monthly volume q."""
    _check_q(q) ## if q is less than 0, it returns an error to the user saying monthly order volume can not be negative
    d, h = 12 * q, annual_holding_cost_per_unit(p)
    eoq = calculate_eoq(d, p.inbound_shipment_fixed_usd, h) if h > 0 else 0.0
    opy = orders_per_year(d, eoq)
    return {"eoq": eoq, "orders_per_year": opy, "days_between": 365 / opy if opy else 0.0,
            "safety_stock": safety_stock(p, q), "reorder_point": reorder_point(p, q)}


def _check_q(q: float) -> None:
    if q < 0:
        raise ModelError("Monthly order volume must be 0 or more.")


def total_cost_direct(p: RegionInputs, q: float) -> float:
    _check_q(q)
    return q * direct_variable_cost(p)


def total_cost_warehouse(p: RegionInputs, q: float) -> float:
    _check_q(q)
    return warehouse_fixed_cost(p) + q * warehouse_variable_cost(p) + cycle_cost_coefficient(p) * sqrt(q)


def break_even_volume(p: RegionInputs) -> Optional[float]:
    """Q* in units/month, or None if the warehouse is never cheaper (v_d <= v_w)."""
    a = direct_variable_cost(p) - warehouse_variable_cost(p)
    if a <= 0:
        return None
    k, f = cycle_cost_coefficient(p), warehouse_fixed_cost(p)
    x = (k + sqrt(k * k + 4 * a * f)) / (2 * a)
    return x * x


def decision_threshold(p: RegionInputs, margin: float = 0.10) -> Optional[float]:
    be = break_even_volume(p)
    return None if be is None else be * (1 + margin)


def recommend(p: RegionInputs, q: float, margin: float = 0.10) -> str:
    be = break_even_volume(p)
    if be is None or q < be:
        return "Ship from SA"
    return "Use warehouse" if q >= be * (1 + margin) else "Review: inside margin"


def monthly_saving(p: RegionInputs, q: float) -> float:
    """Direct minus warehouse cost at Q (positive = warehouse cheaper)."""
    return total_cost_direct(p, q) - total_cost_warehouse(p, q)


def cost_curve(p: RegionInputs, q_max: float, points: int = 25) -> List[Dict[str, float]]:
    step = q_max / (points - 1)
    return [{"q": round(i * step, 1), "direct": total_cost_direct(p, i * step),
             "warehouse": total_cost_warehouse(p, i * step)} for i in range(points)]


def apply_scenario(p: RegionInputs, freight_pct=0.0, customs_pct=0.0, lease_pct=0.0, fx_pct=0.0) -> RegionInputs:
    """What-if shocks as fractions (0.10 = +10%). fx_pct > 0 means the rand weakens."""
    return replace(
        p,
        direct_freight_usd=p.direct_freight_usd * (1 + freight_pct),
        inbound_freight_usd=p.inbound_freight_usd * (1 + freight_pct),
        direct_customs_ins_usd=p.direct_customs_ins_usd * (1 + customs_pct),
        inbound_customs_ins_usd=p.inbound_customs_ins_usd * (1 + customs_pct),
        rent_usd_month=p.rent_usd_month * (1 + lease_pct),
        fx_zar_per_usd=p.fx_zar_per_usd * (1 + fx_pct),
    )


def sensitivity(p: RegionInputs, shock: float = 0.10) -> List[Dict]:
    """Break-even when each input moves by -shock / +shock; sorted by swing."""
    rows = []
    for f in fields(p):
        base = getattr(p, f.name)
        if base == 0:
            continue
        lo = break_even_volume(replace(p, **{f.name: base * (1 - shock)}))
        hi = break_even_volume(replace(p, **{f.name: base * (1 + shock)}))
        swing = abs(hi - lo) if lo is not None and hi is not None else float("inf")
        rows.append({"name": f.name, "label": f.metadata["label"], "low": lo, "high": hi, "swing": swing})
    return sorted(rows, key=lambda r: r["swing"], reverse=True)


def progress_to_break_even(q: float, be: Optional[float]) -> Optional[float]:
    return None if not be else q / be

