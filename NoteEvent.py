

class Event:
    """
    A single musical event in notation terms.
    - pitch: MIDI int (0–127) or string like "C#4" or None for rest
    - length: 'whole','half','quarter','eighth','sixteenth',
              'dotted_half','dotted_quarter','dotted_eighth',
              'triplet_half','triplet_quarter','triplet_eighth'
    - velocity: 1–127 (ignored for rests)
    - articulation: None | 'staccato' | 'legato' | 'tenuto' | 'accent'
    - start_offset_beats: float offset from the running cursor (syncopation)
    - tie_next: if True and next event same pitch, merge duration
    """
    __slots__ = ("pitch","beats","velocity","articulation",
                 "start_offset_beats","tie_next")
    def __init__(self, pitch=None, beats=1.0, velocity=100,
                articulation=None, start_offset_beats=0.0, tie_next=False):
        self.pitch = self._to_midi(pitch) if pitch is not None else None
        self.beats = float(beats)              # ONLY beats, no length names
        self.velocity = int(velocity)
        self.articulation = articulation
        self.start_offset_beats = float(start_offset_beats)
        self.tie_next = bool(tie_next)

    # --- helpers ---
    @staticmethod
    def _to_midi(p):
        if isinstance(p, int):
            return p
        m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", str(p).strip())
        if not m:
            raise ValueError(f"Bad pitch: {p}")
        name, acc, octv = m.groups()
        base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[name.upper()]
        if acc == "#": base += 1
        if acc == "b": base -= 1
        return base + (int(octv)+1)*12  # MIDI C4=60

    @staticmethod
    def apply_articulation(beats: float, articulation: str|None) -> float:
        if articulation == "staccato":
            return beats*0.5
        if articulation == "tenuto":
            return beats*0.95
        if articulation == "legato":
            return beats*1.05  # small overlap feel
        return beats  # accent handled as velocity, duration unchanged



class InstrumentTrack:
    """
    Holds a list of Events and renders them.
    - program: GM program number (e.g., 81 = Lead 2 Saw)
    - bpm, ppq, time_signature
    - start_offset_beats: track-level delay
    """
    def __init__(self, program=0, channel=0, bpm=120, ppq=480,
                 time_signature=(4,4), start_offset_beats=0.0):
        self.program = int(program)
        self.channel = int(channel)
        self.bpm = float(bpm)
        self.ppq = int(ppq)
        self.time_signature = tuple(time_signature)
        self.start_offset_beats = float(start_offset_beats)
        self.events: list[Event] = []

        # File paths
        self.midi_file = "track.mid"
        self.wav_raw = "track_raw.wav"
        self.soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

        # Simple DSP defaults
        self.eq = {"low_cut": None, "high_cut": None, "peak": None}  # peak=(freq, gain_db, Q)
        self.comp = {"threshold_db": None, "ratio": 4}
        self.reverb = {"decay": 0.0, "delay_s": 0.08}

    # --- builder API ---
    def add(self, event: Event):
        self.events.append(event)
        return self
    def add_note(self, pitch, beats=1.0, **kwargs):
        return self.add(Event(pitch=pitch, beats=beats, **kwargs))

    def add_rest(self, beats=1.0, **kwargs):
        return self.add(Event(pitch=None, beats=beats, **kwargs))


    def to_note_sequence(self) -> music_pb2.NoteSequence:
        seq = music_pb2.NoteSequence()
        seq.tempos.add(qpm=self.bpm)

        cursor_beats = self.start_offset_beats
        i = 0
        while i < len(self.events):
            ev = self.events[i]

            # Get beats directly (no conversion needed)
            beats = ev.beats

            # Handle tied notes - merge beats into a single duration
            if ev.pitch is not None and ev.tie_next:
                total_beats = beats
                k = i + 1
                while k < len(self.events):
                    nxt = self.events[k]
                    if nxt.pitch is None or nxt.pitch != ev.pitch:
                        break
                    total_beats += nxt.beats
                    k += 1
                    if not nxt.tie_next:  # Last in tie chain
                        break
                beats = total_beats
                skip_until = k
            else:
                skip_until = i

            # Handle rests
            if ev.pitch is None:
                cursor_beats += beats + ev.start_offset_beats
                i += 1
                continue

            # Apply articulation to sounding duration
            sound_beats = beats
            if ev.articulation == "staccato":
                sound_beats = beats * 0.5
            elif ev.articulation == "tenuto":
                sound_beats = beats * 0.95
            elif ev.articulation == "legato":
                sound_beats = beats * 1.05

            # Calculate start position with offset
            start_beats = cursor_beats + ev.start_offset_beats

            # Convert beats to seconds (quarter = 60/bpm seconds)
            beat_sec = 60.0 / self.bpm
            start_time = start_beats * beat_sec
            end_time = start_time + sound_beats * beat_sec

            # Create note in sequence
            note = seq.notes.add()
            note.pitch = ev.pitch
            note.start_time = start_time
            note.end_time = end_time
            note.velocity = min(127, max(1, int(ev.velocity)))
            note.program = self.program
            note.instrument = self.channel

            # Advance cursor by musical beats (not shortened sound)
            cursor_beats = start_beats + beats

            # Skip tied notes if any
            i = skip_until + 1

        seq.total_time = cursor_beats * (60.0 / self.bpm)
        return seq



    # --- core ---
    def write_midi(self, path=None):
        if path: self.midi_file = path
        seq = self.to_note_sequence()
        note_seq.sequence_proto_to_midi_file(seq, self.midi_file)
        print(f"[MIDI] wrote {self.midi_file}")

    def render(self, soundfont=None, wav_out=None, sr=44100):
        if soundfont: self.soundfont_path = soundfont
        if wav_out: self.wav_raw = wav_out
        subprocess.run([
            "fluidsynth",
            "-ni", self.soundfont_path,
            self.midi_file,
            "-F", self.wav_raw,
            "-r", str(sr)
        ], check=True)
        print(f"[Render] wrote {self.wav_raw}")
    def print_track_info(self):
        """Display track information."""
        print(f"\n=== Track Info ===")
        print(f"Program: {self.program}, BPM: {self.bpm}")
        print(f"Events: {len(self.events)}")

        total_beats = sum(event.beats for event in self.events)
        total_seconds = total_beats * (60.0 / self.bpm)
        print(f"Duration: {total_beats}beats ({total_seconds:.2f}s)")

        print(f"\nEvent Details:")
        for i, event in enumerate(self.events):
            pitch_str = "rest" if event.pitch is None else f"MIDI{event.pitch}"
            print(f"  {i+1}: {pitch_str}, {event.beats}beats, vel:{event.velocity}")
import json
import re
from typing import List, Dict, Any



def convert_json_to_audio_track(complete_events: List[Dict[str, Any]],
                               orchestration: Dict[str, Any]) -> InstrumentTrack:
    """
    Convert LLM-generated JSON data to InstrumentTrack for audio generation.

    Args:
        complete_events: List of event dicts with pitch, beats, velocity, etc.
        orchestration: Dict with bpm, instrument, etc.

    Returns:
        InstrumentTrack ready for audio generation
    """
    instrument_mapping = {
        'piano': 1,                    # Acoustic Grand Piano
        'electric piano': 5,           # Electric Piano 1
        'organ': 17,                   # Drawbar Organ
        'synthesizer': 81,             # Lead 1 (Square)
        'synth': 81,                   # Lead 1 (Square)

        'guitar': 25,                  # Electric Guitar (Clean)
        'electric guitar': 25,         # Electric Guitar (Clean)
        'acoustic guitar': 25,         # Electric Guitar (Clean)
        'distorted guitar': 29,        # Distorted Guitar
        'rock guitar': 29,             # Distorted Guitar

        'bass': 34,                    # Electric Bass (Finger)
        'electric bass': 34,           # Electric Bass (Finger)
        'synth bass': 39,              # Synth Bass 1

        'violin': 41,                  # Violin
        'strings': 49,                 # String Ensemble 1

        'trumpet': 57,                 # Trumpet
        'saxophone': 67,               # Tenor Sax
        'sax': 67,                     # Tenor Sax
        'flute': 74,                   # Flute

        'drums': 128,                  # Drum Kit (channel 10)
    }


    # Get program number from instrument
    instrument = orchestration.get('instrument', 'piano').lower()
    program = instrument_mapping.get(instrument, 1)  # Default to piano

    # Create InstrumentTrack
    track = InstrumentTrack(
        program=program,
        bpm=orchestration.get('bpm', 120),
        channel=0
    )

    # Convert each event
    for event_data in complete_events:
        if event_data['pitch'] == 'rest':
            track.add_rest(
                beats=event_data['beats'],
                start_offset_beats=event_data.get('start_offset_beats', 0.0),
                tie_next=event_data.get('tie_next', False)
            )
        else:
            track.add_note(
                pitch=event_data['pitch'],
                beats=event_data['beats'],
                velocity=event_data['velocity'],
                articulation=event_data.get('articulation'),
                start_offset_beats=event_data.get('start_offset_beats', 0.0),
                tie_next=event_data.get('tie_next', False)
            )


    return track

def create_audio_files(events, orchestration, filename_base):
    """
    Simple alternative to FluidSynth - just create MIDI and skip WAV for now.
    Your existing system already creates MIDI files properly.
    """
    
    print(f"🎵 Creating files for: {filename_base}")
    
    try:
        # Method 1: Use your existing MIDI system
        audio_track = convert_json_to_audio_track(events, orchestration)
        
        # Create MIDI file (this already works in your system)
        midi_filename = f"{filename_base}.mid"
        audio_track.write_midi(midi_filename)
        
        # Calculate duration for info
        total_beats = sum(event['beats'] for event in events)
        bpm = orchestration.get('bpm', 120)
        duration = total_beats * (60.0 / bpm)
        
        print(f"✅ Created MIDI: {midi_filename}")
        print(f"📊 Duration: {duration:.2f} seconds")
        print(f"🎹 Instrument: {orchestration.get('instrument', 'piano')}")
        print(f"🎼 BPM: {bpm}")
        
        return {
            'midi_file': midi_filename,
            'wav_file': None,  
            'duration': duration
        }
        
    except Exception as e:
        print(f"❌ File creation failed: {e}")
        return None


