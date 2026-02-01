import numpy as np
from TradingEnv import TradingEnv

env1 = TradingEnv(seed=42)
env2 = TradingEnv(seed=42)

obs1, _ = env1.reset()
obs2, _ = env2.reset()

assert np.allclose(obs1, obs2), 

# Step both envs with same actions
for _ in range(50):
    action = env1.action_space.sample()
    obs1, _, done1, _, _ = env1.step(action)
    obs2, _, done2, _, _ = env2.step(action)

    assert np.allclose(obs1, obs2), 
    assert done1 == done2, 

print("Environment is deterministic")
