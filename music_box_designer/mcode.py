import base64
import math
import re
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Literal, NamedTuple, Self, TextIO

import mido
from mido import Message, MidiFile, MidiTrack
from PIL import Image, ImageDraw

from .consts import DEFAULT_DURATION, MIDI_DEFAULT_TICKS_PER_BEAT
from .log import logger
from .presets import MusicBox, music_box_presets

DEFAULT_PPQ: int = 96
DEFAULT_PUNCHER_TIMES: int = 2
DEFAULT_START_Y = 0
DEFAULT_END_Y = 1000
base64_regex: str = r'([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)'

ADD_NUMBER: dict[Literal[15, 20, 30], int] = {
    15: 101,
    20: 201,
    30: 1,
}


class MCodeMessage(NamedTuple):
    M: int
    Y: int
    P: int = DEFAULT_PUNCHER_TIMES

    def __str__(self) -> str:
        return f'M{self.M} Y{self.Y} P{self.P}'

    @classmethod
    def from_str(cls, s: str) -> Self:
        try:
            Mxx, Yxx, Pxx = s.strip().split()
            if Mxx[0] != 'M' or Yxx[0] != 'Y' or Pxx[0] != 'P':
                raise ValueError
            M = int(Mxx[1:])
            Y = int(Yxx[1:])
            P = int(Pxx[1:])
        except Exception:
            raise ValueError(f'Invalid mcode message: {s}')
        return cls(M, Y, P)


class MCodeComment(str):
    pass


class MCodeNote(NamedTuple):
    pitch_index: int
    tick: int


def calculate_distance(delta_index: int,
                       delta_tick: int,
                       ppq: int = DEFAULT_PPQ,
                       note_count: Literal[15, 20, 30] = 30) -> float:
    return math.hypot(delta_index * music_box_presets[note_count].grid_width,
                      delta_tick / ppq * music_box_presets[note_count].length_mm_per_beat)


class _NoteLine(NamedTuple):
    """For dynamic programming. A note line is a line of notes with the same tick."""
    pitch_indexes: list[int]  # Should not be empty
    tick: int


def _get_note_lines(notes: list[MCodeNote]) -> list[_NoteLine]:
    note_lines: list[_NoteLine] = []
    i: int = 0
    while i < len(notes):
        tick: int = notes[i].tick
        pitch_indexes: list[int] = []
        while (i < len(notes)
               and notes[i].tick == tick):
            pitch_indexes.append(notes[i].pitch_index)
            i += 1
        note_lines.append(_NoteLine(pitch_indexes=pitch_indexes, tick=tick))
    return note_lines


def get_arranged_notes(notes: list[MCodeNote],
                       ppq: int = DEFAULT_PPQ,
                       note_count: Literal[15, 20, 30] = 30) -> list[MCodeNote]:
    # Do we have an algorithm which uses O(1) extra space?
    notes = sorted(notes, key=lambda note: (note.tick, note.pitch_index))
    note_lines: list[_NoteLine] = _get_note_lines(notes)

    distance_positive: float = 0  # Positive means from lowest to highest
    distance_negative: float = 0  # Negative means from highest to lowest
    route_positive: list[bool] = []  # True for positive, False for negative
    route_negative: list[bool] = []
    for i in range(len(note_lines)):
        previous_note_line = note_lines[i - 1] if i > 0 else note_lines[0]
        current_note_line = note_lines[i]
        # If we merge these lines in a for loop, the code would be more concise but less readable.
        distance_positive_positive: float = distance_positive + calculate_distance(
            previous_note_line.pitch_indexes[-1] - current_note_line.pitch_indexes[0],
            previous_note_line.tick - current_note_line.tick,
            ppq, note_count,
        )
        distance_negative_positive: float = distance_negative + calculate_distance(
            previous_note_line.pitch_indexes[0] - current_note_line.pitch_indexes[0],
            previous_note_line.tick - current_note_line.tick,
            ppq, note_count,
        )
        distance_positive_negative: float = distance_positive + calculate_distance(
            previous_note_line.pitch_indexes[-1] - current_note_line.pitch_indexes[-1],
            previous_note_line.tick - current_note_line.tick,
            ppq, note_count,
        )
        distance_negative_negative: float = distance_negative + calculate_distance(
            previous_note_line.pitch_indexes[0] - current_note_line.pitch_indexes[-1],
            previous_note_line.tick - current_note_line.tick,
            ppq, note_count,
        )

        distance_positive = min(distance_positive_positive, distance_negative_positive)
        route_positive.append(distance_positive_positive < distance_negative_positive)
        distance_negative = min(distance_positive_negative, distance_negative_negative)
        route_negative.append(distance_positive_negative < distance_negative_negative)

        # It doesn't matter whether current_line_distance is added or not.
        current_line_distance: float = calculate_distance(
            0,
            current_note_line.pitch_indexes[-1] - current_note_line.pitch_indexes[0],
            ppq, note_count,
        )
        distance_positive += current_line_distance
        distance_negative += current_line_distance

    final_direction: bool = distance_positive < distance_negative
    route_reversed: list[bool] = []
    current_direction: bool = final_direction
    for i in reversed(range(len(note_lines))):
        # Work backwards to find the route.
        # route_positive, route_negative, note_lines have the same length.
        # route_positive[0] and route_negative[0] are meaningless.
        route_reversed.append(current_direction)
        current_direction = route_positive[i] if current_direction else route_negative[i]

    notes_arranged: list[MCodeNote] = []
    for note_line, direction in zip(note_lines, reversed(route_reversed)):
        if direction:
            notes_arranged.extend(
                MCodeNote(pitch_index=pitch_index, tick=note_line.tick)
                for pitch_index in note_line.pitch_indexes
            )
        else:
            notes_arranged.extend(
                MCodeNote(pitch_index=pitch_index, tick=note_line.tick)
                for pitch_index in reversed(note_line.pitch_indexes)
            )
    return notes_arranged


def notes_to_messages(notes: Iterable[MCodeNote],
                      puncher_times: int = DEFAULT_PUNCHER_TIMES) -> Generator[MCodeMessage, None, None]:
    tick: int = 0
    for note in notes:
        yield MCodeMessage(note.pitch_index, note.tick - tick, puncher_times)
        tick = note.tick


def messages_to_notes(messages: Iterable[MCodeMessage],
                      ignore_M90_M80_Y: bool = True) -> Generator[MCodeNote, None, None]:
    tick: int = 0
    for message in messages:
        if message.M not in (90, 80) or not ignore_M90_M80_Y:
            tick += message.Y
        if message.M in (90, 80):
            continue
        yield MCodeNote(pitch_index=message.M, tick=tick)


@dataclass
class MCodeFile:
    ppq: int = DEFAULT_PPQ
    note_count: Literal[15, 20, 30] = 30
    puncher_times: int = DEFAULT_PUNCHER_TIMES
    messages: list[MCodeMessage] = field(default_factory=list)
    comments: list[str] = field(default_factory=lambda: [''] * 5)

    @classmethod
    def open(cls, file: str | Path | TextIO) -> Self:
        if isinstance(file, str | Path):
            with open(file, 'r', encoding='utf-8') as fp:
                return cls.from_str(fp.read())
        else:
            return cls.from_str(file.read())

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls.from_lines(s.splitlines())

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        mcode_file: Self = cls(comments=[])
        for line in lines:
            if not line:
                continue
            if line.startswith('//'):
                mcode_file.comments.append(line[2:])
            else:
                try:
                    Mxx, Yxx, Pxx = line.strip().split()
                    if Mxx[0] != 'M' or Yxx[0] != 'Y' or Pxx[0] != 'P':
                        raise ValueError
                    M = int(Mxx[1:])
                    Y = int(Yxx[1:])
                    P = int(Pxx[1:])
                    mcode_file.messages.append(MCodeMessage(M, Y, P))
                except Exception:
                    raise ValueError(f'Invalid line: {line}')
        return mcode_file

    @classmethod
    def from_midi(cls,
                  midi_file: MidiFile,
                  note_count: Literal[15, 20, 30, None] = None,
                  transposition: int = 0,
                  puncher_times: int = DEFAULT_PUNCHER_TIMES,
                  store_bytes: bool = True) -> Self:
        # If note_count is None, choose the best one that can contain most notes.
        if note_count is not None:
            actual_note_count: Literal[15, 20, 30] = note_count
        else:
            note_pitches: list[int] = [
                message.note
                for track in midi_file.tracks for message in track
                if message.type == 'note_on'
            ]
            out_of_range_note_number: dict[Literal[15, 20, 30], int] = {
                _note_count:
                sum(1 for note_pitch in note_pitches
                    if note_pitch + transposition not in music_box_presets[_note_count].range)
                for _note_count in (15, 20, 30)
            }
            actual_note_count = min(out_of_range_note_number, key=out_of_range_note_number.get)  # type: ignore

        mcode_file: Self = cls(puncher_times=puncher_times, note_count=actual_note_count)
        preset: MusicBox = music_box_presets[actual_note_count]
        notes: list[MCodeNote] = []
        for track in midi_file.tracks:
            midi_tick: int = 0
            for message in track:
                midi_tick += message.time
                if message.type == 'note_on' and message.velocity > 0:
                    try:
                        pitch_index: int = preset.range.index(message.note + transposition)
                    except ValueError:
                        logger.warning(f'Note {message.note + transposition} is not in the range of the music box.')
                        continue
                    notes.append(MCodeNote(pitch_index + ADD_NUMBER[actual_note_count],
                                           round(midi_tick / midi_file.ticks_per_beat * mcode_file.ppq)))

        notes = get_arranged_notes(notes, mcode_file.ppq, mcode_file.note_count)

        mcode_file.messages.append(MCodeMessage(90, DEFAULT_START_Y, 0))
        mcode_file.messages.extend(notes_to_messages(notes, puncher_times))
        mcode_file.messages.append(MCodeMessage(80, DEFAULT_END_Y, 0))

        total_ticks: int = notes[-1].tick if notes else 0
        mcode_file.comments[0] = f'Total: {total_ticks} ticks'
        mcode_file.comments[1] = f'PPQ: {mcode_file.ppq} ticks'
        mcode_file.comments[2] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        mcode_file.comments[3] = f'MusicBoxPuncher MCode. VERSION 1.4. Generated by Music Box Designer.'
        if store_bytes:
            bytes_io = BytesIO()
            midi_file.save(file=bytes_io)
            bytes_data: bytes = bytes_io.getvalue()
            base64_str: str = base64.b64encode(bytes_data).decode('utf-8')
            mcode_file.comments[4] = f'MIDI {base64_str}'
        else:
            mcode_file.comments[4] = ''

        return mcode_file

    def export_midi(self,
                    use_comment: bool = True,
                    transposition: int = 0,
                    ticks_per_beat: int = MIDI_DEFAULT_TICKS_PER_BEAT) -> MidiFile:
        midi_file: MidiFile = MidiFile()
        midi_file.ticks_per_beat = ticks_per_beat
        midi_track: MidiTrack = MidiTrack()

        if use_comment:
            match = re.search(rf'MIDI ({base64_regex})', self.comments[4])
            if match is None:
                logger.warning('No midi data found in comments.')
            else:
                base64_str: str = match.group(1)
                bytes_data: bytes = base64.b64decode(base64_str)
                midi_file = MidiFile(file=BytesIO(bytes_data))
                return midi_file

        for pitch_index, tick in messages_to_notes(self.messages):
            pitch: int = music_box_presets[self.note_count].range[pitch_index - ADD_NUMBER[self.note_count]] + transposition
            if pitch not in range(128):
                logger.warning(f'Note {pitch} is not in range(128).')
                continue
            midi_track.append(Message('note_on',
                                      note=pitch,
                                      velocity=64,
                                      time=round(tick / self.ppq * ticks_per_beat)))
            midi_track.append(Message('note_off',
                                      note=pitch,
                                      time=round(((tick / self.ppq) + DEFAULT_DURATION) * ticks_per_beat)))
        midi_track.sort(key=lambda message: message.time)
        midi_file.tracks.append(MidiTrack(mido.midifiles.tracks._to_reltime(midi_track)))

        return midi_file

    def generate_pic(self, ppi: float = 300) -> Image.Image:
        from .draft import draw_circle, mm_to_pixel, pos_mm_to_pixel

        preset: MusicBox = music_box_presets[self.note_count]
        notes: list[MCodeNote] = list(messages_to_notes(self.messages, ignore_M90_M80_Y=False))
        tick: int = notes[-1].tick if notes else 0
        length: float = tick / self.ppq * preset.length_mm_per_beat
        image_size: tuple[int, int] = pos_mm_to_pixel((preset.col_width, length), ppi, 'round')
        image: Image.Image = Image.new('RGBA', image_size, 'white')
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
        for index, tick in notes:
            draw_circle(
                image,
                pos_mm_to_pixel((preset.left_border + index * preset.grid_width,
                                 tick / self.ppq * preset.length_mm_per_beat),
                                ppi, 'round'),
                mm_to_pixel(1, ppi), 'black',
            )
        for (index0, tick0), (index1, tick1) in pairwise(notes):
            draw.line(
                (pos_mm_to_pixel((preset.left_border + index0 * preset.grid_width,
                                  tick0 / self.ppq * preset.length_mm_per_beat),
                                 ppi, 'round'),
                 pos_mm_to_pixel((preset.left_border + index1 * preset.grid_width,
                                  tick1 / self.ppq * preset.length_mm_per_beat),
                                 ppi, 'round')),
                'black',
                round(mm_to_pixel(0.5, ppi)),
            )
        return image

    def iter_lines(self) -> Generator[str, None, None]:
        for message in self.messages:
            yield str(message)
        for comment in self.comments:
            yield f'//{comment}'

    def __str__(self) -> str:
        return '\n'.join(self.iter_lines())

    def save(self, file: str | Path | TextIO) -> None:
        # if len(self.comments) != 5:
        #     raise ValueError(f'Length of comments should be 5, got {len(self.comments)}.')
        s: str = str(self)
        if isinstance(file, str | Path):
            with open(file, 'w', encoding='utf-8') as fp:
                fp.write(s)
        else:
            file.write(s)
