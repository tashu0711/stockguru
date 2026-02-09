from django.db import models
from django.utils import timezone

# ── Paper Trading Models ──────────────────────────────────────────────

class PaperTrader(models.Model):
    """Session-based virtual trader with ₹1 Lakh starting balance."""
    session_key = models.CharField(max_length=40, unique=True, db_index=True)
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=100000.00)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Trader {self.session_key[:8]} - ₹{self.balance}"


class Trade(models.Model):
    """Record of each buy/sell transaction."""
    BUY = "BUY"
    SELL = "SELL"
    ACTION_CHOICES = [(BUY, "Buy"), (SELL, "Sell")]

    trader = models.ForeignKey(PaperTrader, on_delete=models.CASCADE, related_name="trades")
    ticker = models.CharField(max_length=20)
    action = models.CharField(max_length=4, choices=ACTION_CHOICES)
    quantity = models.IntegerField()
    price = models.DecimalField(max_digits=12, decimal_places=2)
    total = models.DecimalField(max_digits=12, decimal_places=2)  # qty * price
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.action} {self.quantity} {self.ticker} @ ₹{self.price}"


class Holding(models.Model):
    """Current stock holdings with average price."""
    trader = models.ForeignKey(PaperTrader, on_delete=models.CASCADE, related_name="holdings")
    ticker = models.CharField(max_length=20)
    quantity = models.IntegerField()
    avg_price = models.DecimalField(max_digits=12, decimal_places=2)

    class Meta:
        unique_together = ("trader", "ticker")

    def __str__(self):
        return f"{self.ticker}: {self.quantity} @ ₹{self.avg_price}"


class SuggestionLog(models.Model):
    """Track screener suggestions and their outcome (for live performance tracking)."""
    ticker = models.CharField(max_length=20)
    suggested_at = models.DateTimeField(auto_now_add=True)
    buy_price = models.DecimalField(max_digits=12, decimal_places=2)
    target_price = models.DecimalField(max_digits=12, decimal_places=2)
    stop_loss = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)

    # Outcome tracking (filled after 1 week)
    outcome_checked = models.BooleanField(default=False)
    hit_target = models.BooleanField(null=True, blank=True)
    exit_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    return_pct = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)

    class Meta:
        ordering = ["-suggested_at"]

    def __str__(self):
        status = "✓" if self.hit_target else "✗" if self.outcome_checked else "⏳"
        return f"{status} {self.ticker} @ ₹{self.buy_price}"
