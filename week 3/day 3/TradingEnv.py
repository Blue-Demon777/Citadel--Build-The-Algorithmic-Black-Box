import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
from order_book import OrderBook 
from market_config import MarketConfig 
from engine import MarketEngine 
from environment import MarketEnvironment 
from logger import Logger 
from events import OrderSubmissionEvent 
from order import Order
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=200,
        book_depth=5,
        max_inventory=20,
        max_cash=100_000,
        transaction_cost=0.01,
        lambda_risk=0.001,          #  configurable λ
        seed=42,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.book_depth = book_depth
        self.max_inventory = max_inventory
        self.max_cash = max_cash
        self.transaction_cost = transaction_cost
        self.lambda_risk = lambda_risk

        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)

        obs_dim = 2 * book_depth + 2 * book_depth + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._build_market()

    def _build_market(self):
        self.book = OrderBook()
        self.logger = Logger()
        self.config = MarketConfig(snapshot_interval=1.0)

        self.engine = MarketEngine(self.book, self.logger)
        self.env = MarketEnvironment(self.engine, self.config)

        self.inventory = 0
        self.cash = float(self.max_cash)

        self.prev_value = self.cash
        self.peak_value = self.cash   # track peak

        self.step_count = 0

    def _get_mid_price(self):
        snap = self.book.current_snapshot()
        bid, ask = snap.best_bid(), snap.best_ask()
        return 0.5 * (bid + ask) if bid and ask else 100.0
    
    def _normalize_obs(self):

        snap = self.book.current_snapshot()
        bids = snap.bids[:self.book_depth]
        asks = snap.asks[:self.book_depth]

        bid_prices = [p for p, q in bids]
        bid_sizes  = [q for p, q in bids]
        ask_prices = [p for p, q in asks]
        ask_sizes  = [q for p, q in asks]

        mid_price = self._get_mid_price()

        norm_bid_prices = [(mid_price - p) / mid_price for p in bid_prices]
        norm_bid_sizes  = [s / self.max_cash for s in bid_sizes]

        norm_ask_prices = [(p - mid_price) / mid_price for p in ask_prices]
        norm_ask_sizes  = [s / self.max_cash for s in ask_sizes]

        while len(norm_bid_prices) < self.book_depth:
            norm_bid_prices.append(0.0)
            norm_bid_sizes.append(0.0)

        while len(norm_ask_prices) < self.book_depth:
            norm_ask_prices.append(0.0)
            norm_ask_sizes.append(0.0)

        norm_inventory = (self.inventory + self.max_inventory) / (2 * self.max_inventory)
        norm_cash = self.cash / self.max_cash

        obs = np.array(
            norm_bid_prices
            + norm_bid_sizes
            + norm_ask_prices
            + norm_ask_sizes
            + [norm_inventory, norm_cash],
            dtype=np.float32,
        )

        return np.clip(obs, 0.0, 1.0)
    
    def calculate_reward(self, mid_price, action):
        value = self.cash + self.inventory * mid_price
        delta_value = value - self.prev_value
        transaction_penalty = self.transaction_cost * (action != 0)
        self.peak_value = max(self.peak_value, value)
        drawdown = max(0.0, self.peak_value - value)
        reward = delta_value - transaction_penalty - self.lambda_risk * drawdown
        self.prev_value = value
        return reward, value, drawdown

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._build_market()
        obs = self._normalize_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        terminated, truncated = False, False

        mid = self._get_mid_price()

        if action == 1 and self.cash >= mid:
            self.engine.order_book.submit(
                Order(f"agent-{self.step_count}", "BUY", None, 1, self.engine.time)
            )
            self.inventory += 1
            self.cash -= mid

        elif action == 2 and self.inventory > 0:
            self.engine.order_book.submit(
                Order(f"agent-{self.step_count}", "SELL", None, 1, self.engine.time)
            )
            self.inventory -= 1
            self.cash += mid

        reward, value, drawdown = self.calculate_reward(mid, action)

        if self.step_count >= self.max_steps:
            truncated = True
        if abs(self.inventory) > self.max_inventory or self.cash <= 0:
            terminated = True

        obs = self._normalize_obs()

        info = {
            "portfolio_value": value,
            "drawdown": drawdown,
            "inventory": self.inventory,
            "cash": self.cash,
        }

        return obs, reward, terminated, truncated, info
