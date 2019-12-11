from .online_history import OnlineHistoryBuffer
from .replay_history import ReplayHistoryBuffer
from .prioritized_replay_history import PrioritizedReplayHistoryBuffer
from .replay_history_extended import ReplayHistoryExtendedBuffer

def get_types():
    return {
        "online": OnlineHistoryBuffer,
        "replay": ReplayHistoryBuffer,
        "prioritized_replay": PrioritizedReplayHistoryBuffer,
        "replay_extended": ReplayHistoryExtendedBuffer
    }