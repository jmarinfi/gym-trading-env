import numpy as np

def custom_reward(history):
    safe_log = np.vectorize(lambda x: np.log(float(x)) if x > 0 else 0)
    step = history['step'][-1]

    # Crecimiento logarítmico del portafolio
    portfolio_return = safe_log(history['portfolio_valuation', -1] / history['portfolio_valuation', -2]) if step > 1 else 0

    # Rendimiento relativo al mercado
    market_return = safe_log(history['data_close', -1] / history['data_close', -2]) if step > 1 else 0
    relative_return = portfolio_return - market_return

    # Volatilidad del portafolio en los últimos 30 registros
    portfolio_returns = np.diff(safe_log(history['portfolio_valuation'][-30:])) if step > 29 else np.diff(safe_log(history['portfolio_valuation']))
    portfolio_volatility = np.std(portfolio_returns)

    # Drawdown
    peak = np.max(history['portfolio_valuation'][-30:]) if step > 29 else np.max(history['portfolio_valuation'])
    current_drawdown = (peak - history['portfolio_valuation', -1]) / peak

    # Utilización del margen
    margin_utilization = abs(history['real_position', -1] - 1)

    # Cálculo de la recompensa final
    reward = (2 * relative_return - 0.5 * portfolio_volatility - current_drawdown - 0.1 * margin_utilization) * 100

    return reward

def simple_custom_reward(history):
    return 800 * np.log(history['portfolio_valuation', -1] / history['portfolio_valuation', -2])
