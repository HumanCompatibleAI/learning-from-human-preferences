from stable_baselines.a2c.a2c import A2C, A2CRunner


class DRLHPActorCritic(A2C):

    # TODO reimplement __init__ in a way that asserts things about the passed-in env
    def _make_runner(self) -> A2CRunner:
        return A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)


class DRLHPRunner(A2CRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to collect data from which a DRLHP policy can be trained. This assumes an environment whereby
        the reward signal passed back is generated from a reward prediction model inside of the env.

        :param env: (Gym environment) The environment to learn from, which needs to implement update_segment_buffer to take in
        new segments to request human preferences about
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(DRLHPRunner, self).__init__(env=env, model=model, n_steps=n_steps, gamma=gamma)
        assert self.env.hasattr('update_segment_buffer'), "Passed-in environment must be wrapped with HumanPreferencesEnvWrapper"


    def run(self):
        mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards = super(DRLHPRunner, self).run()
        # TODO fix update_segment_buffer so that it takes in masks instead of dones
        self.env.update_segment_buffer(mb_obs, mb_rewards, mb_masks)
