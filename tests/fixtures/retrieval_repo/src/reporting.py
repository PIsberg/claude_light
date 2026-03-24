def generate_weekly_sales_report(rows: list[dict]) -> dict[str, float]:
    total_orders = len(rows)
    total_revenue = sum(row["amount"] for row in rows)
    average_order_value = total_revenue / total_orders if total_orders else 0.0
    return {
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "average_order_value": round(average_order_value, 2),
    }


def render_report_summary(stats: dict[str, float]) -> str:
    return (
        f"orders={stats['total_orders']} "
        f"revenue={stats['total_revenue']:.2f} "
        f"aov={stats['average_order_value']:.2f}"
    )
