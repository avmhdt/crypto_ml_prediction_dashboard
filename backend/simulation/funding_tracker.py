"""Funding rate tracker for perpetual futures simulation.

Tracks 8-hour (standard) or 4-hour funding intervals and computes
the funding cost for positions held across funding timestamps.
Binance perpetual futures charge/pay funding every 8 hours at
00:00, 08:00, 16:00 UTC (or every 4 hours for some contracts).
"""


# Standard 8-hour funding times (UTC hours)
FUNDING_TIMES_8H: list[int] = [0, 8, 16]

# Alternative 4-hour funding times (UTC hours)
FUNDING_TIMES_4H: list[int] = [0, 4, 8, 12, 16, 20]

# Milliseconds per hour
_MS_PER_HOUR = 3_600_000


class FundingRateTracker:
    """Track funding intervals and compute funding costs.

    Args:
        default_rate: Default funding rate per interval (e.g. 0.0001
            = 1 bps per 8h period).  Positive rate means longs pay
            shorts.
        funding_interval_hours: Hours between funding events (8 or 4).
    """

    def __init__(
        self,
        default_rate: float = 0.0001,
        funding_interval_hours: int = 8,
    ) -> None:
        self._current_rate = default_rate
        self._next_funding_ms: int = 0
        self._funding_interval_hours = funding_interval_hours

        if funding_interval_hours == 4:
            self._funding_times = FUNDING_TIMES_4H
        else:
            self._funding_times = FUNDING_TIMES_8H

    # --- rate updates --------------------------------------------------

    def set_rate(self, rate: float, next_funding_ms: int) -> None:
        """Update the current funding rate and next funding timestamp.

        Args:
            rate: New funding rate (positive = longs pay shorts).
            next_funding_ms: Next funding event timestamp in epoch ms.
        """
        self._current_rate = rate
        self._next_funding_ms = next_funding_ms

    # --- cost computation ----------------------------------------------

    def compute_funding_cost(
        self,
        entry_ms: int,
        exit_ms: int,
        position_notional: float,
        side: int,
    ) -> float:
        """Compute total funding cost for a position held over a period.

        Counts how many funding timestamps fall between entry and exit,
        then multiplies by rate and notional.

        Sign convention:
        - If ``side == 1`` (long) and ``rate > 0``: cost is positive
          (long pays).
        - If ``side == -1`` (short) and ``rate > 0``: cost is negative
          (short receives, so net cost is negative = rebate).
        - Returns the net cost: positive means the user pays.

        Args:
            entry_ms: Position entry timestamp in epoch milliseconds.
            exit_ms: Position exit timestamp in epoch milliseconds.
            position_notional: Absolute notional value of the position.
            side: ``1`` for long, ``-1`` for short.

        Returns:
            Net funding cost in USD.  Positive = user pays,
            negative = user receives.
        """
        if entry_ms >= exit_ms or position_notional <= 0.0:
            return 0.0

        num_events = self._count_funding_events(entry_ms, exit_ms)
        if num_events == 0:
            return 0.0

        # Long pays when rate > 0; short receives when rate > 0
        cost = num_events * self._current_rate * position_notional
        return cost * side

    def _count_funding_events(
        self,
        entry_ms: int,
        exit_ms: int,
    ) -> int:
        """Count how many funding timestamps fall in (entry, exit].

        Converts millisecond timestamps to UTC hours and counts how
        many scheduled funding times are crossed.

        Args:
            entry_ms: Start of period (exclusive) in epoch ms.
            exit_ms: End of period (inclusive) in epoch ms.

        Returns:
            Number of funding events in the interval.
        """
        if entry_ms >= exit_ms:
            return 0

        # Convert to total hours since epoch
        entry_hour = entry_ms / _MS_PER_HOUR
        exit_hour = exit_ms / _MS_PER_HOUR

        # Total number of hours spanned
        total_hours = exit_hour - entry_hour

        # Number of complete days spanned
        events_per_day = len(self._funding_times)
        complete_days = int(total_hours // 24)
        events = complete_days * events_per_day

        # Handle the partial days at the boundaries
        # Get UTC hour-of-day for entry and exit
        entry_utc_hour = entry_hour % 24
        remaining_hours = total_hours - (complete_days * 24)
        exit_utc_hour_equiv = entry_utc_hour + remaining_hours

        for ft in self._funding_times:
            # Check if this funding time falls in the partial window
            # We need to check ft and ft+24 to handle wrap-around
            for offset in [0, 24]:
                adjusted_ft = ft + offset
                if entry_utc_hour < adjusted_ft <= exit_utc_hour_equiv:
                    events += 1

        return events

    # --- properties ----------------------------------------------------

    @property
    def current_rate(self) -> float:
        """Current funding rate per interval."""
        return self._current_rate

    @property
    def next_funding_ms(self) -> int:
        """Next funding event timestamp in epoch milliseconds."""
        return self._next_funding_ms

    @property
    def funding_interval_hours(self) -> int:
        """Hours between funding events."""
        return self._funding_interval_hours
