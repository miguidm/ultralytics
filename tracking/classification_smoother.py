#!/usr/bin/env python3
"""
Classification Smoother for Object Tracking

Reduces classification flickering (e.g., car ↔ truck) by applying temporal smoothing
to maintain stable class assignments across frames.

Strategies:
1. majority_vote: Use most common class in sliding window
2. confidence_weighted: Weight votes by detection confidence
3. lock_after_n: Lock class after N consistent frames
4. hysteresis: Require sustained evidence before switching
"""

from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np


@dataclass
class TrackClassHistory:
    """Stores classification history for a single track."""
    track_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    locked_class: Optional[str] = None
    lock_frame: int = 0
    frames_since_last_switch: int = 0
    last_class: Optional[str] = None
    switch_count: int = 0


class ClassificationSmoother:
    """
    Smooths classification outputs to reduce flickering between similar classes.

    Usage:
        smoother = ClassificationSmoother(strategy='majority_vote', window_size=10)

        # In tracking loop:
        for detection in detections:
            raw_class = model.names[class_id]
            stable_class = smoother.get_stable_class(
                track_id=obj_id,
                raw_class=raw_class,
                confidence=conf,
                frame_num=frame_count
            )
    """

    def __init__(
        self,
        strategy: str = 'majority_vote',
        window_size: int = 10,
        lock_threshold: int = 15,
        switch_threshold: int = 5,
        confidence_boost: float = 0.1,
        min_confidence_to_switch: float = 0.6
    ):
        """
        Initialize the classification smoother.

        Args:
            strategy: Smoothing strategy ('majority_vote', 'confidence_weighted',
                     'lock_after_n', 'hysteresis')
            window_size: Number of frames for sliding window (majority_vote, confidence_weighted)
            lock_threshold: Frames before locking class (lock_after_n)
            switch_threshold: Consecutive frames of new class needed to switch (hysteresis)
            confidence_boost: Extra confidence weight for current class (confidence_weighted)
            min_confidence_to_switch: Minimum confidence to consider switching (hysteresis)
        """
        self.strategy = strategy
        self.window_size = window_size
        self.lock_threshold = lock_threshold
        self.switch_threshold = switch_threshold
        self.confidence_boost = confidence_boost
        self.min_confidence_to_switch = min_confidence_to_switch

        # Track histories
        self.tracks: Dict[int, TrackClassHistory] = {}

        # Statistics
        self.total_queries = 0
        self.switches_prevented = 0
        self.raw_switches = 0
        self.smoothed_switches = 0

    def _get_or_create_track(self, track_id: int) -> TrackClassHistory:
        """Get existing track history or create new one."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackClassHistory(
                track_id=track_id,
                history=deque(maxlen=self.window_size),
                confidence_history=deque(maxlen=self.window_size)
            )
        return self.tracks[track_id]

    def get_stable_class(
        self,
        track_id: int,
        raw_class: str,
        confidence: float = 1.0,
        frame_num: int = 0
    ) -> str:
        """
        Get smoothed/stable classification for a track.

        Args:
            track_id: Unique track identifier
            raw_class: Raw class from detector
            confidence: Detection confidence (0-1)
            frame_num: Current frame number

        Returns:
            Stable class name after applying smoothing
        """
        self.total_queries += 1
        track = self._get_or_create_track(track_id)

        # Record raw switch
        if track.last_class is not None and raw_class != track.last_class:
            self.raw_switches += 1

        # Add to history
        track.history.append(raw_class)
        track.confidence_history.append(confidence)

        # Apply smoothing strategy
        if self.strategy == 'majority_vote':
            stable_class = self._majority_vote(track)
        elif self.strategy == 'confidence_weighted':
            stable_class = self._confidence_weighted(track)
        elif self.strategy == 'lock_after_n':
            stable_class = self._lock_after_n(track, frame_num)
        elif self.strategy == 'hysteresis':
            stable_class = self._hysteresis(track, raw_class, confidence)
        else:
            stable_class = raw_class  # Fallback to raw

        # Track smoothed switches
        if track.last_class is not None and stable_class != track.last_class:
            self.smoothed_switches += 1
            track.switch_count += 1
            track.frames_since_last_switch = 0
        else:
            track.frames_since_last_switch += 1

        # Update switches prevented
        if raw_class != stable_class:
            self.switches_prevented += 1

        track.last_class = stable_class
        return stable_class

    def _majority_vote(self, track: TrackClassHistory) -> str:
        """Return most common class in the window."""
        if not track.history:
            return "unknown"

        counts = Counter(track.history)
        return counts.most_common(1)[0][0]

    def _confidence_weighted(self, track: TrackClassHistory) -> str:
        """Return class with highest confidence-weighted votes."""
        if not track.history:
            return "unknown"

        # Weight each class by its confidence
        class_weights = defaultdict(float)
        for cls, conf in zip(track.history, track.confidence_history):
            class_weights[cls] += conf

        # Boost current class to reduce flickering
        if track.last_class and track.last_class in class_weights:
            class_weights[track.last_class] += self.confidence_boost * len(track.history)

        return max(class_weights, key=class_weights.get)

    def _lock_after_n(self, track: TrackClassHistory, frame_num: int) -> str:
        """Lock class after N frames of tracking."""
        if not track.history:
            return "unknown"

        # If already locked, return locked class
        if track.locked_class is not None:
            return track.locked_class

        # Check if we should lock
        if len(track.history) >= self.lock_threshold:
            # Lock to majority class
            counts = Counter(track.history)
            track.locked_class = counts.most_common(1)[0][0]
            track.lock_frame = frame_num
            return track.locked_class

        # Not yet locked - use majority vote
        return self._majority_vote(track)

    def _hysteresis(self, track: TrackClassHistory, raw_class: str, confidence: float) -> str:
        """Require sustained evidence before switching class."""
        if not track.history:
            track.last_class = raw_class
            return raw_class

        current_class = track.last_class or raw_class

        # Count recent consecutive occurrences of raw_class
        consecutive_new = 0
        for cls in reversed(track.history):
            if cls == raw_class:
                consecutive_new += 1
            else:
                break

        # Only switch if:
        # 1. New class has been consistent for switch_threshold frames
        # 2. Confidence is above minimum
        if (raw_class != current_class and
            consecutive_new >= self.switch_threshold and
            confidence >= self.min_confidence_to_switch):
            return raw_class

        return current_class

    def get_statistics(self) -> Dict:
        """Return smoothing statistics."""
        return {
            'total_queries': self.total_queries,
            'raw_switches': self.raw_switches,
            'smoothed_switches': self.smoothed_switches,
            'switches_prevented': self.switches_prevented,
            'reduction_pct': (
                100 * (self.raw_switches - self.smoothed_switches) / max(1, self.raw_switches)
            ),
            'active_tracks': len(self.tracks),
            'tracks_with_switches': sum(1 for t in self.tracks.values() if t.switch_count > 0)
        }

    def reset_track(self, track_id: int):
        """Reset history for a specific track."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def cleanup_old_tracks(self, active_track_ids: set):
        """Remove tracks that are no longer active."""
        to_remove = [tid for tid in self.tracks if tid not in active_track_ids]
        for tid in to_remove:
            del self.tracks[tid]


def create_smoother(strategy: str = 'hysteresis', **kwargs) -> ClassificationSmoother:
    """
    Factory function to create a smoother with recommended defaults.

    Recommended strategies:
    - 'majority_vote': Good general purpose, window_size=10
    - 'hysteresis': Best for reducing flickering, switch_threshold=5
    - 'lock_after_n': Best for stable objects, lock_threshold=15
    """
    defaults = {
        'majority_vote': {'window_size': 10},
        'confidence_weighted': {'window_size': 10, 'confidence_boost': 0.15},
        'lock_after_n': {'lock_threshold': 15},
        'hysteresis': {'switch_threshold': 5, 'min_confidence_to_switch': 0.55}
    }

    params = defaults.get(strategy, {})
    params.update(kwargs)

    return ClassificationSmoother(strategy=strategy, **params)


# Demo / test
if __name__ == "__main__":
    print("Testing ClassificationSmoother...")
    print()

    # Simulate flickering track
    smoother = create_smoother('hysteresis', switch_threshold=3)

    # Simulated sequence: mostly car with brief truck flickers
    sequence = [
        ('car', 0.85), ('car', 0.82), ('car', 0.88),
        ('truck', 0.51),  # flicker
        ('car', 0.86), ('car', 0.84),
        ('truck', 0.48),  # flicker
        ('truck', 0.52),  # flicker
        ('car', 0.87), ('car', 0.89), ('car', 0.85),
        ('truck', 0.72), ('truck', 0.75), ('truck', 0.78), ('truck', 0.80),  # genuine switch
        ('truck', 0.79), ('truck', 0.81)
    ]

    print("Frame | Raw Class | Confidence | Stable Class")
    print("-" * 50)

    for i, (raw_class, conf) in enumerate(sequence):
        stable = smoother.get_stable_class(
            track_id=1,
            raw_class=raw_class,
            confidence=conf,
            frame_num=i
        )
        changed = " <-- SWITCH" if i > 0 and stable != sequence[i-1][0] and raw_class == stable else ""
        prevented = " (prevented)" if raw_class != stable else ""
        print(f"  {i:2d}  | {raw_class:9s} | {conf:.2f}       | {stable:12s}{prevented}{changed}")

    print()
    print("Statistics:")
    stats = smoother.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
