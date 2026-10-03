from collections.abc import Sequence
from dataclasses import dataclass

from .keyboard import BACKSPACE

PHRASES: tuple[str, ...] = (
    "THE SKY IS BLUE",
    "WE LIKE COLD TEA",
    "OPEN THE DOOR",
    "SEE YOU AT NOON",
    "BIG DOGS RUN FAST",
)


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class PhraseResult:
    phrase: str
    typed: str
    seconds: float
    selections: int
    backspaces: int

    @property
    def wpm(self) -> float:
        return 12.0 * len(self.typed) / self.seconds if self.seconds > 0 else 0.0

    @property
    def error_rate(self) -> float:
        denominator = max(len(self.phrase), len(self.typed))
        return edit_distance(self.phrase, self.typed) / denominator if denominator else 0.0

    @property
    def keystrokes_per_char(self) -> float:
        return self.selections / len(self.typed) if self.typed else 0.0


class StudyController:
    def __init__(self, phrases: Sequence[str] = PHRASES) -> None:
        if not phrases or any(not p for p in phrases):
            raise ValueError("need at least one non-empty phrase")
        self._phrases = tuple(phrases)
        self._index = 0
        self.results: list[PhraseResult] = []
        self._reset()

    def _reset(self) -> None:
        self._t_first: int | None = None
        self._t_last = 0
        self._selections = 0
        self._backspaces = 0

    @property
    def n_phrases(self) -> int:
        return len(self._phrases)

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._index >= len(self._phrases)

    @property
    def current_phrase(self) -> str | None:
        return None if self.finished else self._phrases[self._index]

    def on_press(self, t_ns: int, key_id: str, text: str) -> bool:
        phrase = self.current_phrase
        if phrase is None:
            raise RuntimeError("the study is already finished")
        if self._t_first is None:
            self._t_first = t_ns
        self._t_last = t_ns
        self._selections += 1
        if key_id == f"c:{BACKSPACE}":
            self._backspaces += 1
        if len(text) < len(phrase):
            return False
        self.results.append(
            PhraseResult(
                phrase=phrase,
                typed=text,
                seconds=(self._t_last - self._t_first) / 1e9,
                selections=self._selections,
                backspaces=self._backspaces,
            )
        )
        self._index += 1
        self._reset()
        return True


@dataclass(frozen=True, slots=True)
class StudySummary:
    n_phrases: int
    wpm: float
    error_rate: float
    keystrokes_per_char: float
    backspaces: int
    total_seconds: float


def summarise_study(results: Sequence[PhraseResult]) -> StudySummary:
    if not results:
        raise ValueError("no results")
    chars = sum(len(r.typed) for r in results)
    seconds = sum(r.seconds for r in results)
    return StudySummary(
        n_phrases=len(results),
        wpm=12.0 * chars / seconds if seconds > 0 else 0.0,
        error_rate=sum(r.error_rate for r in results) / len(results),
        keystrokes_per_char=sum(r.selections for r in results) / chars if chars else 0.0,
        backspaces=sum(r.backspaces for r in results),
        total_seconds=seconds,
    )
