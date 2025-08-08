from note_seq.protobuf import music_pb2

# ===============================
# 1) Event: one note or rest
# ===============================
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
    __slots__ = ("pitch","length","velocity","articulation",
                 "start_offset_beats","tie_next")

    def __init__(self, pitch=None, length="quarter", velocity=100,
                 articulation=None, start_offset_beats=0.0, tie_next=False):
        self.pitch = self._to_midi(pitch) if pitch is not None else None
        self.length = length
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
    def beats_for_length(token: str) -> float:
        base = {
            "whole":4.0, "half":2.0, "quarter":1.0,
            "eighth":0.5, "sixteenth":0.25
        }
        if token in base:
            return base[token]
        if token.startswith("dotted_"):
            raw = token.replace("dotted_","")
            return base[raw]*1.5
        if token.startswith("triplet_"):
            raw = token.replace("triplet_","")
            return base[raw]*(2.0/3.0)
        raise ValueError(f"Unknown length token: {token}")

    @staticmethod
    def apply_articulation(beats: float, articulation: str|None) -> float:
        if articulation == "staccato":
            return beats*0.5
        if articulation == "tenuto":
            return beats*0.95
        if articulation == "legato":
            return beats*1.05  # small overlap feel
        return beats  # accent handled as velocity, duration unchanged


# ==================================
# 2) InstrumentTrack: timeline + IO
# ==================================
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
        self.wav_processed = "track_processed.wav"
        self.soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

        # Simple DSP defaults
        self.eq = {"low_cut": None, "high_cut": None, "peak": None}  # peak=(freq, gain_db, Q)
        self.comp = {"threshold_db": None, "ratio": 4}
        self.reverb = {"decay": 0.0, "delay_s": 0.08}

    # --- builder API ---
    def add(self, event: Event):
        self.events.append(event)
        return self

    def add_note(self, pitch, length="quarter", **kwargs):
        return self.add(Event(pitch=pitch, length=length, **kwargs))

    def add_rest(self, length="quarter", **kwargs):
        return self.add(Event(pitch=None, length=length, **kwargs))

    # --- core ---
    def to_note_sequence(self) -> music_pb2.NoteSequence:
        seq = music_pb2.NoteSequence()
        seq.tempos.add(qpm=self.bpm)

        cursor_beats = self.start_offset_beats
        i = 0
        while i < len(self.events):
            ev = self.events[i]
            # compute musical length in beats
            beats = Event.beats_for_length(ev.length)
            # merge ties into a single duration
            j = i
            if ev.pitch is not None and ev.tie_next:
                total = beats
                k = i+1
                while k < len(self.events):
                    nxt = self.events[k]
                    if nxt.pitch is None or nxt.pitch != ev.pitch or not nxt.tie_next and k != i+1 and not nxt.tie_next:
                        # include last if it's same pitch and previous had tie_next True
                        # but if nxt.tie_next is False and pitch same, we still merge this final one
                        break
                    total += Event.beats_for_length(nxt.length)
                    k += 1
                    if not nxt.tie_next:  # last in chain
                        break
                beats = total
                # we will skip the tied followers
                skip_until = k
            else:
                skip_until = i

            # rest
            if ev.pitch is None:
                cursor_beats += beats + ev.start_offset_beats
                i += 1
                continue

            # articulation duration (sounding)
            sound_beats = Event.apply_articulation(beats, ev.articulation)

            # start offset for syncopation
            start_beats = cursor_beats + ev.start_offset_beats

            # convert to seconds (quarter = 60/bpm sec)
            beat_sec = 60.0 / self.bpm
            start = start_beats * beat_sec
            end = start + sound_beats * beat_sec

            note = seq.notes.add()
            note.pitch = ev.pitch
            note.start_time = start
            note.end_time = end
            note.velocity = min(127, max(1, int(ev.velocity)))
            note.program = self.program
            note.instrument = self.channel

            # advance musical grid by *musical* beats (not staccato shortened sound)
            cursor_beats = start_beats + beats

            # skip tied followers if any
            i = skip_until + 1

        seq.total_time = cursor_beats * (60.0 / self.bpm)
        return seq

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

    # ---- optional simple DSP (EQ/comp/reverb) ----
    def process(self, wav_out=None):
        if wav_out: self.wav_processed = wav_out
        y, sr = librosa.load(self.wav_raw, sr=None)

        # EQ
        if self.eq.get("low_cut"):
            b,a = signal.butter(2, self.eq["low_cut"], btype='high', fs=sr)
            y = signal.filtfilt(b,a,y)
        if self.eq.get("high_cut"):
            b,a = signal.butter(2, self.eq["high_cut"], btype='low', fs=sr)
            y = signal.filtfilt(b,a,y)
        if self.eq.get("peak"):
            freq, gain_db, Q = self.eq["peak"]
            A = 10**(gain_db/40)
            w0 = 2*np.pi*freq/sr
            alpha = np.sin(w0)/(2*Q)
            b0 = 1 + alpha*A; b1 = -2*np.cos(w0); b2 = 1 - alpha*A
            a0 = 1 + alpha/A; a1 = -2*np.cos(w0); a2 = 1 - alpha/A
            b = np.array([b0,b1,b2]) / a0
            a = np.array([1,a1/a0,a2/a0])
            y = signal.lfilter(b,a,y)

        # Compression (very simple)
        if self.comp.get("threshold_db") is not None:
            thr = 10**(self.comp["threshold_db"]/20.0)
            ratio = float(self.comp.get("ratio", 4))
            over = np.abs(y) > thr
            y2 = y.copy()
            y2[over] = np.sign(y[over]) * (thr + (np.abs(y[over]) - thr)/ratio)
            y = y2

        # Reverb (simple feedback delay)
        if self.reverb.get("decay", 0.0) > 0:
            decay = float(self.reverb["decay"])
            delay_s = float(self.reverb.get("delay_s", 0.08))
            d = int(sr*delay_s)
            out = y.copy()
            for i in range(d, len(out)):
                out[i] += decay*out[i-d]
            y = out

        sf.write(self.wav_processed, y, sr)
        print(f"[Process] wrote {self.wav_processed}")

    def audio(self):
        return ipd.Audio(self.wav_processed if os.path.exists(self.wav_processed) else self.wav_raw)
