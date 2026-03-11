from backend.bars.time_bars import TimeBars
from backend.bars.information_bars import TickBars, VolumeBars, DollarBars
from backend.bars.imbalance_bars import TickImbalanceBars, VolumeImbalanceBars, DollarImbalanceBars
from backend.bars.run_bars import TickRunBars, VolumeRunBars, DollarRunBars

BAR_CLASSES = {
    "time": TimeBars,
    "tick": TickBars,
    "volume": VolumeBars,
    "dollar": DollarBars,
    "tick_imbalance": TickImbalanceBars,
    "volume_imbalance": VolumeImbalanceBars,
    "dollar_imbalance": DollarImbalanceBars,
    "tick_run": TickRunBars,
    "volume_run": VolumeRunBars,
    "dollar_run": DollarRunBars,
}
