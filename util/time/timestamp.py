import copy
from datetime import datetime, timedelta
from functools import total_ordering
from typing import Optional, List


@total_ordering
class Timestamp:
    """
    A timestamp with an origin time and an optional lead time.
    We call a timestamp a "target timestamp" if it has a lead time, and an "origin timestamp" otherwise.
    """

    def __init__(self, year, month, day, hour, lead_time: int = None):
        """
        - lead_time: the number of hours ahead of the origin time
        """
        self.origin: datetime = datetime(year, month, day, hour)
        self.lead_time: Optional[int] = lead_time

    @property
    def target(self):
        if self.lead_time is None:
            return None
        else:
            return self.origin + timedelta(hours=self.lead_time)

    def __eq__(self, other):
        if not isinstance(other, Timestamp):
            return NotImplemented
        return (self.origin, self.lead_time) == (other.origin, other.lead_time)

    def __lt__(self, other):
        if not isinstance(other, Timestamp):
            return NotImplemented
        return (self.origin, self.lead_time) < (other.origin, other.lead_time)

    def __repr__(self):
        return f"Timestamp(origin={self.origin}, lead_time={self.lead_time})"


def get_timestamp_range(start: Timestamp, end: Timestamp, step=timedelta(hours=1), start_lead_time: int = None,
                        end_lead_time: int = None) -> List[Timestamp]:
    """
    Get a list of timestamps between two timestamps. Start inclusive, end exclusive.

    Examples:
    - Origin timestamps:
      get_timestamps_between(Timestamp(2021, 1, 1, 0, None), Timestamp(2021, 1, 1, 4, None)) yields:
      [
          Timestamp(origin=datetime.datetime(2021, 1, 1, 0), lead_time=None),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 1), lead_time=None),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 2), lead_time=None),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 3), lead_time=None),
      ]
    - Target timestamps:
      get_timestamps_between(Timestamp(2021, 1, 1, 0, 7), Timestamp(2021, 1, 1, 2, 7), start_lead_time=6,
      end_lead_time=9) yields:
      [
          Timestamp(origin=datetime.datetime(2021, 1, 1, 0), lead_time=7),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 0), lead_time=8),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 1), lead_time=6),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 1), lead_time=7),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 1), lead_time=8),
          Timestamp(origin=datetime.datetime(2021, 1, 1, 2), lead_time=6),
      ]
    """
    if start.lead_time is None and end.lead_time is not None or start.lead_time is not None and end.lead_time is None:
        raise ValueError("Both start and end timestamps must be either both origin or target timestamps")

    timestamps = []
    if start.lead_time is None:
        current = start.origin
        while current < end.origin:
            timestamps.append(copy.deepcopy(current))
            current += step
    else:
        if start_lead_time is None or end_lead_time is None:
            raise ValueError("min_lead_time and max_lead_time must be provided for target timestamps")
        if start_lead_time > end_lead_time:
            raise ValueError("min_lead_time must be less than or equal to max_lead_time")

        timestamps = []
        current = start
        while current.origin <= end.origin and current.target < end.target:
            timestamps.append(copy.deepcopy(current))
            current.lead_time += 1
            if current.lead_time >= end_lead_time:
                current.origin += step
                current.lead_time = start_lead_time

    return timestamps
