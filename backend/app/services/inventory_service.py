"""
Inventory intelligence service — simulated stock replay.

This version models inventory explicitly instead of using a single static
"days of stock" heuristic. The simulation replays daily demand with:

- initial stock
- reorder point
- order-up-to target
- lead time
- replenishment arrivals
- lost sales / stockout events

That gives the project a more realistic end-to-end inventory process while
still working with sales-only datasets such as UCI Online Retail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.services import data_store, forecast_service

logger = logging.getLogger(__name__)

_ALERT_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "critical": 1.01,
}


@dataclass
class PendingOrder:
    arrival_date: date
    quantity: float


def _classify_alert(stockout_risk: float) -> str:
    if stockout_risk < _ALERT_THRESHOLDS["low"]:
        return "low"
    if stockout_risk < _ALERT_THRESHOLDS["medium"]:
        return "medium"
    if stockout_risk < _ALERT_THRESHOLDS["high"]:
        return "high"
    return "critical"


def _build_dense_history(stock_code: str, history_days: int) -> pd.DataFrame:
    """Return a dense daily demand series, filling missing days with zero demand."""
    history = data_store.get_product_sales_history(stock_code, last_n=None)
    if history.empty:
        return history

    history = history.copy()
    history["sale_date"] = pd.to_datetime(history["sale_date"]).dt.normalize()
    history = (
        history.groupby("sale_date", as_index=False)["total_quantity"]
        .sum()
        .sort_values("sale_date")
    )

    full_dates = pd.date_range(
        start=history["sale_date"].min(),
        end=history["sale_date"].max(),
        freq="D",
    )
    dense = pd.DataFrame({"sale_date": full_dates}).merge(history, on="sale_date", how="left")
    dense["total_quantity"] = dense["total_quantity"].fillna(0.0).astype(float)
    if history_days > 0:
        dense = dense.tail(history_days)
    return dense.reset_index(drop=True)


def _policy(avg_daily_demand: float, settings: Settings) -> tuple[float, float]:
    """Return ``(reorder_point, target_stock_level)`` for the replenishment policy."""
    reorder_point = avg_daily_demand * (
        settings.INVENTORY_LEAD_TIME_DAYS + settings.INVENTORY_SAFETY_STOCK_DAYS
    )
    target_stock_level = avg_daily_demand * settings.INVENTORY_TARGET_COVER_DAYS
    return float(reorder_point), float(target_stock_level)


def _run_inventory_simulation(
    timeline: list[tuple[date, float]],
    *,
    starting_stock: float,
    reorder_point: float,
    target_stock_level: float,
    avg_daily_demand: float,
    lead_time_days: int,
    pending_orders: list[PendingOrder] | None = None,
) -> tuple[list[dict], float, list[PendingOrder]]:
    """Replay a demand timeline and return per-day inventory states."""
    stock = float(starting_stock)
    backlog = [
        PendingOrder(order.arrival_date, float(order.quantity))
        for order in (pending_orders or [])
    ]
    records: list[dict] = []

    for current_date, demand in timeline:
        received_qty = sum(order.quantity for order in backlog if order.arrival_date <= current_date)
        backlog = [order for order in backlog if order.arrival_date > current_date]
        stock += received_qty

        stock_before_sales = stock
        fulfilled_qty = min(stock_before_sales, demand)
        lost_sales = max(demand - stock_before_sales, 0.0)
        stock = max(stock_before_sales - demand, 0.0)
        stockout = lost_sales > 0.0

        order_qty = 0.0
        arrival_date: date | None = None
        pending_qty = sum(order.quantity for order in backlog)

        if (
            avg_daily_demand > 0
            and stock <= reorder_point
            and pending_qty <= 0
        ):
            order_qty = max(target_stock_level - stock, avg_daily_demand * max(lead_time_days, 1))
            order_qty = round(float(order_qty), 4)
            if order_qty > 0:
                arrival_date = current_date + timedelta(days=lead_time_days)
                backlog.append(PendingOrder(arrival_date=arrival_date, quantity=order_qty))
                pending_qty += order_qty

        records.append(
            {
                "date": current_date,
                "demand": round(float(demand), 4),
                "received_qty": round(float(received_qty), 4),
                "stock_before_sales": round(float(stock_before_sales), 4),
                "fulfilled_qty": round(float(fulfilled_qty), 4),
                "lost_sales": round(float(lost_sales), 4),
                "stock_after_sales": round(float(stock), 4),
                "order_qty": order_qty,
                "arrival_date": arrival_date,
                "pending_qty": round(float(pending_qty), 4),
                "stockout": stockout,
            }
        )

    return records, round(float(stock), 4), backlog


def _estimate_stockout_probability(
    stock_code: str,
    future_forecast: list[dict],
    *,
    starting_stock: float,
    pending_orders: list[PendingOrder],
    reorder_point: float,
    target_stock_level: float,
    avg_daily_demand: float,
    demand_std: float,
    settings: Settings,
) -> float:
    """Monte Carlo stockout probability over the forecast horizon."""
    if not future_forecast:
        return 0.0

    runs = max(settings.INVENTORY_MONTE_CARLO_RUNS, 1)
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(stock_code))
    rng = np.random.default_rng(seed)
    stockout_runs = 0

    for _ in range(runs):
        scenario: list[tuple[date, float]] = []
        for point in future_forecast:
            mean = max(float(point["predicted_quantity"]), 0.0)
            if demand_std <= 1e-6:
                sampled = mean
            elif mean < 1.0:
                sampled = float(rng.poisson(lam=max(mean, 0.0)))
            else:
                sampled = max(0.0, float(rng.normal(loc=mean, scale=demand_std)))
            scenario.append((point["forecast_date"], sampled))

        records, _, _ = _run_inventory_simulation(
            scenario,
            starting_stock=starting_stock,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            avg_daily_demand=avg_daily_demand,
            lead_time_days=settings.INVENTORY_LEAD_TIME_DAYS,
            pending_orders=pending_orders,
        )
        if any(record["stockout"] for record in records):
            stockout_runs += 1

    return round(stockout_runs / runs, 4)


def compute_inventory_signal(stock_code: str) -> dict:
    """Compute inventory intelligence for ``stock_code`` using stock replay."""
    settings = get_settings()

    dense_history = _build_dense_history(stock_code, settings.INVENTORY_SIM_HISTORY_DAYS)
    if dense_history.empty:
        raise KeyError(f"No sales history for stock_code={stock_code!r}")

    product_info = data_store.get_product_info(stock_code)
    description = product_info.get("description") if product_info else None

    recent_28 = dense_history.tail(28)
    recent_56 = dense_history.tail(56)
    avg_daily_demand = float(recent_28["total_quantity"].mean()) if not recent_28.empty else 0.0
    demand_std = float(recent_56["total_quantity"].std(ddof=0)) if len(recent_56) > 1 else 0.0
    avg_daily_demand = max(avg_daily_demand, 0.0)
    demand_std = max(demand_std, 0.0)

    as_of_date: date = dense_history["sale_date"].max().date()
    initial_stock_level = round(avg_daily_demand * settings.INVENTORY_STOCK_MULTIPLIER, 4)
    reorder_point, target_stock_level = _policy(avg_daily_demand, settings)

    history_timeline = [
        (row["sale_date"].date(), float(row["total_quantity"]))
        for _, row in dense_history.iterrows()
    ]
    historical_records, current_stock, pending_orders = _run_inventory_simulation(
        history_timeline,
        starting_stock=initial_stock_level,
        reorder_point=reorder_point,
        target_stock_level=target_stock_level,
        avg_daily_demand=avg_daily_demand,
        lead_time_days=settings.INVENTORY_LEAD_TIME_DAYS,
    )

    try:
        future_forecast = forecast_service.generate_forecast(
            stock_code,
            horizon=max(settings.DEFAULT_FORECAST_HORIZON, settings.INVENTORY_LEAD_TIME_DAYS + 7),
        )
    except Exception:
        future_forecast = []

    predicted_next = (
        float(future_forecast[0]["predicted_quantity"])
        if future_forecast
        else avg_daily_demand
    )

    future_timeline = [
        (point["forecast_date"], float(point["predicted_quantity"]))
        for point in future_forecast
    ]
    projected_records, _, projected_pending = _run_inventory_simulation(
        future_timeline,
        starting_stock=current_stock,
        reorder_point=reorder_point,
        target_stock_level=target_stock_level,
        avg_daily_demand=avg_daily_demand,
        lead_time_days=settings.INVENTORY_LEAD_TIME_DAYS,
        pending_orders=pending_orders,
    )

    stockout_risk = _estimate_stockout_probability(
        stock_code,
        future_forecast,
        starting_stock=current_stock,
        pending_orders=pending_orders,
        reorder_point=reorder_point,
        target_stock_level=target_stock_level,
        avg_daily_demand=avg_daily_demand,
        demand_std=demand_std,
        settings=settings,
    )

    last_30_records = historical_records[-30:]
    stockout_days_last_30 = sum(1 for record in last_30_records if record["stockout"])
    total_demand_last_30 = sum(record["demand"] for record in last_30_records)
    lost_sales_last_30 = sum(record["lost_sales"] for record in last_30_records)
    if total_demand_last_30 > 0:
        service_level_last_30 = max(0.0, 1.0 - lost_sales_last_30 / total_demand_last_30)
    else:
        service_level_last_30 = 1.0

    last_restock_date = None
    for record in reversed(historical_records):
        if record["received_qty"] > 0:
            last_restock_date = record["date"]
            break

    pending_restock_quantity = round(sum(order.quantity for order in projected_pending), 4)
    next_restock_date = min((order.arrival_date for order in projected_pending), default=None)
    projected_stockout_days = sum(1 for record in projected_records if record["stockout"])

    first_projected_stockout = next(
        (record for record in projected_records if record["stockout"]),
        None,
    )
    if first_projected_stockout is not None:
        days_of_stock_remaining = (first_projected_stockout["date"] - as_of_date).days
    elif avg_daily_demand > 0:
        days_of_stock_remaining = int(current_stock / avg_daily_demand)
    else:
        days_of_stock_remaining = 0

    reorder_suggested = current_stock <= reorder_point and pending_restock_quantity <= 0
    alert_level = _classify_alert(stockout_risk)

    logger.debug(
        "Inventory simulation for %s: stock=%.2f reorder_point=%.2f pending=%.2f "
        "risk=%.2f stockout_days_30=%d projected_stockout_days=%d",
        stock_code,
        current_stock,
        reorder_point,
        pending_restock_quantity,
        stockout_risk,
        stockout_days_last_30,
        projected_stockout_days,
    )

    return {
        "stock_code": stock_code,
        "description": description,
        "as_of_date": as_of_date,
        "avg_daily_demand": round(avg_daily_demand, 4),
        "predicted_next_demand": round(float(predicted_next), 4),
        "initial_stock_level": initial_stock_level,
        "target_stock_level": round(target_stock_level, 4),
        "simulated_stock_level": round(float(current_stock), 4),
        "reorder_point": round(float(reorder_point), 4),
        "days_of_stock_remaining": max(days_of_stock_remaining, 0),
        "pending_restock_quantity": pending_restock_quantity,
        "next_restock_date": next_restock_date,
        "last_restock_date": last_restock_date,
        "stockout_days_last_30": int(stockout_days_last_30),
        "projected_stockout_days": int(projected_stockout_days),
        "service_level_last_30": round(float(service_level_last_30), 4),
        "stockout_risk": round(float(stockout_risk), 4),
        "alert_level": alert_level,
        "reorder_suggested": reorder_suggested,
    }
