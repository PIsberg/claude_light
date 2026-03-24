def calculate_invoice_total(subtotal: float, tax_rate: float, discount: float) -> float:
    discounted = subtotal - discount
    taxed = discounted * (1.0 + tax_rate)
    return round(taxed, 2)


def format_invoice_line_item(name: str, amount: float) -> str:
    return f"{name}: ${amount:.2f}"
