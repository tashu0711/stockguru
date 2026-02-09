from django.urls import path
from . import views

urlpatterns = [
    # Original stock analysis
    path('', views.analyze_stock, name='home'),

    # Smart Screener
    path('screener/', views.smart_screener, name='screener'),
    path('backtest/', views.backtest_report, name='backtest'),

    # Paper Trading
    path('portfolio/', views.portfolio, name='portfolio'),
    path('buy/', views.buy_stock, name='buy_stock'),
    path('sell/', views.sell_stock, name='sell_stock'),
    path('history/', views.trade_history, name='trade_history'),
]