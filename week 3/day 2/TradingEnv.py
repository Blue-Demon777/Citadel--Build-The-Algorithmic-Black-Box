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
    """
    Minimal, deterministic trading environment (v1).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=200,
        book_depth=5,
        max_inventory=20,
        max_cash=100_000,
        transaction_cost=0.01,
        seed=42,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.book_depth = book_depth
        self.max_inventory = max_inventory
        self.max_cash = max_cash
        self.transaction_cost = transaction_cost

        self._rng = np.random.default_rng(seed)

        # ----- Action Space -----
        # 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)

        # ----- Observation Space -----
        obs_dim = (
            2 * book_depth          # prices
            + 2 * book_depth        # volumes
            + 2                     # inventory, cash
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._build_market()

    # ---------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------

    def _build_market(self):
        self.book = OrderBook()
        self.logger = Logger()
        self.config = MarketConfig(snapshot_interval=1.0)

        self.engine = MarketEngine(self.book, self.logger)
        self.env = MarketEnvironment(self.engine, self.config)

        self.inventory = 0
        self.cash = float(self.max_cash)
        self.prev_pnl = self.cash

        self.step_count = 0

    def _get_mid_price(self):
        snap = self.book.current_snapshot()
        bid = snap.best_bid()
        ask = snap.best_ask()

        if bid is None or ask is None:
            return 100.0  # stable fallback
        return 0.5 * (bid + ask)

    def _normalize_obs(self):
        snap = self.book.current_snapshot()
        mid = self._get_mid_price()

        bids = snap.bids[: self.book_depth]
        asks = snap.asks[: self.book_depth]

        # Prices normalized by mid
        bid_prices = [(p / mid) for p, _ in bids]
        ask_prices = [(p / mid) for p, _ in asks]

        # Volumes log-scaled
        bid_vols = [np.log1p(q) for _, q in bids]
        ask_vols = [np.log1p(q) for _, q in asks]

        # Padding
        while len(bid_prices) < self.book_depth:
            bid_prices.append(1.0)
            bid_vols.append(0.0)

        while len(ask_prices) < self.book_depth:
            ask_prices.append(1.0)
            ask_vols.append(0.0)

        inventory_norm = (self.inventory + self.max_inventory) / (2 * self.max_inventory)
        cash_norm = self.cash / self.max_cash

        obs = np.array(
            bid_prices + ask_prices + bid_vols + ask_vols + [inventory_norm, cash_norm],
            dtype=np.float32,
        )

        return np.clip(obs, 0.0, 1.0)

    # ---------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._build_market()

        obs = self._normalize_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = False

        mid = self._get_mid_price()

        # ----- Execute Action -----
        if action == 1:  # BUY
            if self.cash >= mid:
                order = Order(
                    order_id=f"agent-{self.step_count}",
                    side="BUY",
                    price=None,
                    qty=1,
                    timestamp=self.engine.time,
                )
                self.engine.order_book.submit(order)
                self.inventory += 1
                self.cash -= mid

        elif action == 2:  # SELL
            if self.inventory > 0:
                order = Order(
                    order_id=f"agent-{self.step_count}",
                    side="SELL",
                    price=None,
                    qty=1,
                    timestamp=self.engine.time,
                )
                self.engine.order_book.submit(order)
                self.inventory -= 1
                self.cash += mid


        # HOLD does nothing

        # ----- Reward -----
        pnl = self.cash + self.inventory * mid
        reward = pnl - self.prev_pnl
        reward -= self.transaction_cost * abs(action != 0)
        self.prev_pnl = pnl

        # ----- Termination Logic -----
        if self.step_count >= self.max_steps:
            truncated = True

        if abs(self.inventory) > self.max_inventory:
            terminated = True

        if self.cash <= 0:
            terminated = True

        obs = self._normalize_obs()

        info = {
            "pnl": pnl,
            "inventory": self.inventory,
            "cash": self.cash,
            "mid_price": mid,
        }

        return obs, reward, terminated, truncated, info
