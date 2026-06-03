from torchrl.envs import EnvBase, RewardSum, TransformedEnv

## environment extensions section
def episode_reward_ext(env: EnvBase, out_keys=[("agents", "episode_reward")]) -> EnvBase:
    return TransformedEnv(
        env,
        RewardSum(
            in_keys=[env.reward_key],
            out_keys=out_keys
        )
    )