import numpy as np

from .replay_history import ReplayHistoryBuffer


class ReplayHistoryExtendedBuffer(ReplayHistoryBuffer):
    def __init__(self, size, train_frequency, avoid_episode_crossing=False, **kwargs):
        super().__init__(
            size,
            train_frequency,
            avoid_episode_crossing=avoid_episode_crossing,
            **kwargs
        )
        self.env_indices = {}

    def _sample_added(self, sample):
        self.env_indices[sample["env_id"]] = sample["info"]["env_index"]
        return super()._sample_added(sample)

    def _make_sample_range(self, env_id, index, steps, fixed_target=False):
        """Prepares a timestep range from an ENV ID for train-batching"""
        assert index >= 0 and (index + steps <= len(self.buffer[env_id]))
        ret = []
        nstep_target = self.nstep_target

        for i in range(index, index + steps):
            # Update nstep discounts/targets (It may already be updated,
            # depending on sub-class, in which case _update_nstep does nothing)
            if fixed_target:
                # In fixed_target mode we never give a target beyond the
                # actual sequence being trained (Common in online training)
                nstep_target = min(nstep_target, index + steps - i)
            self._update_nstep(env_id, i, nstep_target)
            sample = self.buffer[env_id][i]
            ret.append(
                {
                    "target_states": sample["target_state"],
                    "states": sample["state"],
                    "returns": sample["return"],
                    "nsteps": sample["nstep"],
                    "target_masks": sample["target_mask"],
                    "policy_outputs": sample["policy_output"],
                    "env_indices": sample["info"]["env_index"]
                }
            )
        return ret
