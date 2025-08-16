
class Note(BaseModel):
    pitch: str = Field(pattern=r"^([A-G][#b]?[0-8]|rest)$")
    beats: float = Field(ge=0.25, le=2.0)

class MelodyResponse(BaseModel):
    notes: List[Note] = Field(max_length=8)
    explanation: str

class ChordProgressionResponse(BaseModel):
    chords: List[str] = Field(description="Chord names like Cmaj, Amin, F#dim, etc.")
    key: str = Field(description="Key signature like 'C major' or 'A minor'")
    progression_explanation: str

class OrchestrationResponse(BaseModel):
    bpm: int
    melody_description: str
    emotion: str
    style: str
    instrument: str

class VelocityArticulationResponse(BaseModel):
    velocities: List[int] = Field(description="Velocity values 40-127")
    articulations: List[Optional[Literal["staccato", "legato", "tenuto", "accent"]]]

class TimingResponse(BaseModel):
    start_offsets: List[float] = Field(description="Start offset values 0.0-1.0")
    tie_nexts: List[bool] = Field(description="Tie next boolean values")

class EnhancedMusicOrchestrator:
    """Complete merged orchestrator - all functionality in one class."""
    
    def __init__(self, pipe):
        # Core components
        self.pipe = pipe
        self.orchestration = None
        self.chord_progression = []
        self.key = None
        self.running_melody = []
        self.structured_model = outlines.from_transformers(pipe.model, pipe.tokenizer)
        
        # Available instruments
        self.available_instruments = {
            'piano': 1, 'electric piano': 5, 'organ': 17, 'synthesizer': 81, 'synth': 81,
            'guitar': 25, 'electric guitar': 25, 'acoustic guitar': 25, 'distorted guitar': 29, 'rock guitar': 29,
            'bass': 34, 'electric bass': 34, 'synth bass': 39,
            'violin': 41, 'strings': 49,
            'trumpet': 57, 'saxophone': 67, 'sax': 67, 'flute': 74,
            'drums': 128,
        }

    def get_chord_tones(self, chord_name: str, octave: int = 4) -> List[str]:
        """Get chord tones using music21 harmony module."""
        # CLEAN FIRST - before any music21 calls
        cleaned_chord = self._clean_chord_name(chord_name)
        
        # Try manual fallback FIRST for known problem chords
        fallback_result = self._manual_chord_fallback(chord_name, octave)
        if fallback_result:
            return fallback_result
        
        try:
            # Now try music21 with cleaned chord name
            chord_symbol = harmony.ChordSymbol(cleaned_chord)
            note_names = [pitch.name for pitch in chord_symbol.pitches]
            return [f"{note}{octave}" for note in note_names]
        except Exception as e:
            print(f"⚠️  Error parsing chord '{chord_name}' (cleaned: '{cleaned_chord}'): {e}")
            return [f"C{octave}", f"E{octave}", f"G{octave}"]
    
    def _clean_chord_name(self, chord_name: str) -> str:
        """Clean up LLM-generated chord names for music21 compatibility."""
        
        # Remove parentheses: E7(#9) → E7#9
        cleaned = chord_name.replace('(', '').replace(')', '')
        
        # Handle common LLM naming variations
        naming_fixes = {
            'Cminor': 'Cm',
            'Dminor': 'Dm', 
            'Eminor': 'Em',
            'Fminor': 'Fm',
            'Gminor': 'Gm',
            'Aminor': 'Am',
            'Bminor': 'Bm',
            'Cmajor': 'C',
            'Dmajor': 'D',
            'Emajor': 'E', 
            'Fmajor': 'F',
            'Gmajor': 'G',
            'Amajor': 'A',
            'Bmajor': 'B',
        }
        
        # Apply naming fixes
        if cleaned in naming_fixes:
            cleaned = naming_fixes[cleaned]
        
        # Handle flat note names that confuse music21
        flat_to_sharp = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
        }
        
        # Check if this is a flat chord causing issues
        for flat_note, sharp_note in flat_to_sharp.items():
            if cleaned.startswith(flat_note):
                # Replace the flat note with sharp equivalent
                cleaned = sharp_note + cleaned[len(flat_note):]
                break
        
        return cleaned.strip()
        replacements = {
            'min': 'm',           # min → m
            'add9': 'add9',       # Keep as is
            'sus2': 'sus2',       # Keep as is
            'sus4': 'sus4',       # Keep as is
        }
        
        # Apply safe replacements
        for old, new in replacements.items():
            if old in cleaned and old != new:
                cleaned = cleaned.replace(old, new)
        
        return cleaned.strip()
    
    def _manual_chord_fallback(self, chord_name: str, octave: int) -> Optional[List[str]]:
        """Manual fallback for chords music21 can't parse."""
        
        # Common problem chords that music21 struggles with
        manual_chords = {
            # Sharp 9 chords
            'E7#9': ['E', 'G#', 'B', 'D', 'F##'],
            'E7(#9)': ['E', 'G#', 'B', 'D', 'F##'],
            'C7#9': ['C', 'E', 'G', 'Bb', 'D#'],
            'C7(#9)': ['C', 'E', 'G', 'Bb', 'D#'],
            'G7#9': ['G', 'B', 'D', 'F', 'A#'],
            'G7(#9)': ['G', 'B', 'D', 'F', 'A#'],
            'A7#9': ['A', 'C#', 'E', 'G', 'B#'],
            'A7(#9)': ['A', 'C#', 'E', 'G', 'B#'],
            'D7#9': ['D', 'F#', 'A', 'C', 'E#'],
            'D7(#9)': ['D', 'F#', 'A', 'C', 'E#'],
            'F7#9': ['F', 'A', 'C', 'Eb', 'G#'],
            'F7(#9)': ['F', 'A', 'C', 'Eb', 'G#'],
            'B7#9': ['B', 'D#', 'F#', 'A', 'C##'],
            'B7(#9)': ['B', 'D#', 'F#', 'A', 'C##'],
            
            # Flat chords that cause issues
            'Ebmaj7': ['Eb', 'G', 'Bb', 'D'],
            'Eb7': ['Eb', 'G', 'Bb', 'Db'],
            'Ebm7': ['Eb', 'Gb', 'Bb', 'Db'],
            'Dbmaj7': ['Db', 'F', 'Ab', 'C'],
            'Db7': ['Db', 'F', 'Ab', 'B'],
            'Dbm7': ['Db', 'E', 'Ab', 'B'],
            'Abmaj7': ['Ab', 'C', 'Eb', 'G'],
            'Ab7': ['Ab', 'C', 'Eb', 'Gb'],
            'Abm7': ['Ab', 'B', 'Eb', 'Gb'],
            'Bbmaj7': ['Bb', 'D', 'F', 'A'],
            'Bb7': ['Bb', 'D', 'F', 'Ab'],
            'Bbm7': ['Bb', 'Db', 'F', 'Ab'],
            'Gbmaj7': ['Gb', 'Bb', 'Db', 'F'],
            'Gb7': ['Gb', 'Bb', 'Db', 'E'],
            'Gbm7': ['Gb', 'A', 'Db', 'E'],
            
            # Power chords with flats
            'Eb5': ['Eb', 'Bb'],
            'Db5': ['Db', 'Ab'], 
            'Ab5': ['Ab', 'Eb'],
            'Bb5': ['Bb', 'F'],
            'Gb5': ['Gb', 'Db'],
        }
        
        if chord_name in manual_chords:
            notes = manual_chords[chord_name]
            print(f"🔧 Using manual fallback for {chord_name}: {notes}")
            return [f"{note}{octave}" for note in notes]
        
        return None

    def orchestrate(self, user_input: str, num_phrases: int = 4, max_new_tokens: int = 300, temperature: float = 0.3):
        """Generate musical framework and chord progression."""
        
        instrument_list = ", ".join(self.available_instruments.keys())

        orchestration_prompt = f"""Convert this music request into specifications.

Request: "{user_input}"

Requirements:
- bpm: tempo between 60-180 BPM (slow: 60-90, medium: 90-130, fast: 130-180)
- emotion: single word (happy, sad, calm, energetic, mysterious, etc.)  # 
- melody_description: Provide a detailed description of the specific melodic techniques, compositional style, and musical characteristics that should be used. Include information about rhythm patterns, melodic motion, harmonic approach, and any stylistic ornamentations or techniques typical of this musical style.
- style: music genre (rock, jazz, classical, electronic, pop, blues, etc.)
- instrument: MUST choose from these options only: {instrument_list}

Choose the instrument that best fits the musical style and user request."""

        try:
            orchestration_result = self.structured_model(
                orchestration_prompt,
                OrchestrationResponse,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            if isinstance(orchestration_result, str):
                self.orchestration = json.loads(orchestration_result)
            else:
                self.orchestration = {
                    'bpm': orchestration_result.bpm,
                    'melody_description': orchestration_result.melody_description,
                    'emotion': orchestration_result.emotion,
                    'style': orchestration_result.style,
                    'instrument': orchestration_result.instrument
                }

        except Exception as e:
            print(f"Orchestration failed: {e}")
            self.orchestration = {
                'bpm': 120, 'melody_description': 'simple melody',
                'emotion': 'neutral', 'style': 'general', 'instrument': 'piano'
            }

        self._generate_chord_progression(num_phrases)

        return {
            'orchestration': self.orchestration,
            'chord_progression': self.chord_progression,
            'key': self.key
        }  # ← FIXED: Removed the random 'z'

    def _generate_chord_progression(self, num_phrases: int):
        """Generate chord progression."""
        
        emotion = self.orchestration['emotion']
        style = self.orchestration['style']

        chord_prompt = f"""Generate a {num_phrases}-chord progression for a {emotion} {style} song.

Requirements:
- Generate exactly {num_phrases} chords (one for each melodic phrase)
- Use standard chord notation: Cmaj7, Am7, F#dim, G7, etc.
- IMPORTANT: Use sharp notes (F#, C#, G#, D#, A#) instead of flat notes (Gb, Db, Ab, Eb, Bb)
- Use these chord formats only: maj7, m7, 7, m, dim, aug, sus2, sus4, 5
- Examples: C#maj7, F#m7, A#7, G#m, D#dim (NO flats like Bb, Eb, Ab)
- Choose a key signature that fits {emotion} {style} music
- Make the progression musically satisfying and style-appropriate

Create a progression that supports {emotion} melodies in {style} style using SHARP-BASED chord names only."""

        try:
            chord_result = self.structured_model(
                chord_prompt,
                ChordProgressionResponse,
                max_new_tokens=200,
                temperature=0.3
            )

            if isinstance(chord_result, str):
                chord_data = json.loads(chord_result)
                self.chord_progression = chord_data['chords']
                self.key = chord_data['key']
                progression_explanation = chord_data.get('progression_explanation', '')
            else:
                self.chord_progression = chord_result.chords
                self.key = chord_result.key
                progression_explanation = chord_result.progression_explanation

            # Ensure correct number of chords
            if len(self.chord_progression) != num_phrases:
                print(f"⚠️ Adjusting chord count: got {len(self.chord_progression)}, need {num_phrases}")
                if len(self.chord_progression) < num_phrases:
                    while len(self.chord_progression) < num_phrases:
                        self.chord_progression.extend(self.chord_progression[:num_phrases - len(self.chord_progression)])
                else:
                    self.chord_progression = self.chord_progression[:num_phrases]

            print(f"✅ Generated chord progression: {' | '.join(self.chord_progression)}")
            print(f"   Key: {self.key}")
            if progression_explanation:
                print(f"   Explanation: {progression_explanation}")

        except Exception as e:
            print(f"Chord progression generation failed: {e}")
            self.chord_progression = self._get_fallback_progression(num_phrases, style)
            self.key = "C major"

    def _get_fallback_progression(self, num_phrases: int, style: str) -> List[str]:
        """Generate fallback chord progressions."""
        
        style_progressions = {
            'pop': ['Cmaj', 'Amin', 'Fmaj', 'Gmaj'],
            'rock': ['Emin', 'Cmaj', 'Gmaj', 'Dmaj'],
            'jazz': ['Cmaj7', 'A7', 'Dmin7', 'G7'],
            'blues': ['C7', 'F7', 'G7', 'C7'],
            'electronic': ['Amin', 'Fmaj', 'Cmaj', 'Gmaj'],
            'classical': ['Cmaj', 'Gmaj', 'Amin', 'Fmaj']
        }

        base_progression = style_progressions.get(style.lower(), ['Cmaj', 'Fmaj', 'Gmaj', 'Cmaj'])
        result = []
        for i in range(num_phrases):
            result.append(base_progression[i % len(base_progression)])
        return result

    def analyze_chord_targeting(self, phrase_notes: List[dict], chord_name: str) -> dict:
        """Analyze how well a phrase targets its chord."""
        chord_tone_names = [note[:-1] for note in self.get_chord_tones(chord_name)]
        
        total_notes = len([n for n in phrase_notes if n['pitch'] != 'rest'])
        if total_notes == 0:
            return {'chord_tone_percentage': 0, 'analysis': 'No notes generated'}
        
        chord_tone_count = 0
        for note in phrase_notes:
            if note['pitch'] != 'rest':
                note_name = note['pitch'][:-1]  # Remove octave
                if note_name in chord_tone_names:
                    chord_tone_count += 1
        
        percentage = (chord_tone_count / total_notes) * 100
        return {
            'chord_tone_percentage': percentage,
            'chord_tone_count': chord_tone_count,
            'total_notes': total_notes,
            'analysis': f"{percentage:.1f}% chord tones ({chord_tone_count}/{total_notes})"
        }

    def analyze_melody_targeting(self):
        """Analyze chord targeting for the entire melody."""
        if not self.running_melody:
            print("No melody to analyze")
            return

        print(f"\n🎯 MELODY CHORD TARGETING ANALYSIS")
        print("=" * 40)

        total_notes = 0
        total_chord_tones = 0

        for i, phrase in enumerate(self.running_melody, 1):
            if 'targeting_analysis' in phrase:
                analysis = phrase['targeting_analysis']
                print(f"Phrase {i} [{phrase['chord']}]: {analysis['analysis']}")
                total_notes += analysis.get('total_notes', len(phrase['notes']))
                total_chord_tones += analysis.get('chord_tone_count', 0)

        if total_notes > 0:
            overall_percentage = (total_chord_tones / total_notes) * 100
            print(f"\n🎼 OVERALL: {overall_percentage:.1f}% chord tones ({total_chord_tones}/{total_notes})")
            
            if overall_percentage >= 70:
                print("✅ Excellent chord targeting!")
            elif overall_percentage >= 50:
                print("⚠️  Moderate chord targeting")
            else:
                print("❌ Poor chord targeting")

        return {
            'total_notes': total_notes,
            'chord_tone_count': total_chord_tones,
            'percentage': overall_percentage if total_notes > 0 else 0
        }

    def melody_generator(self, max_new_tokens: int = 400, temperature: float = 0.4, use_targeting: bool = True, max_retries: int = 3):
        """Enhanced melody generator with chord targeting."""
        
        if not self.orchestration or not self.chord_progression:
            raise ValueError("Must call orchestrate() first")

        phrase_number = len(self.running_melody)
        if phrase_number >= len(self.chord_progression):
            raise ValueError(f"No more chords available.")

        current_chord = self.chord_progression[phrase_number]
        
        # Get parameters
        bpm = self.orchestration["bpm"]
        melody_description = self.orchestration["melody_description"]
        emotion = self.orchestration["emotion"]
        style = self.orchestration["style"]

        # Build context
        if self.running_melody:
            all_notes_so_far = []
            for phrase in self.running_melody:
                all_notes_so_far.extend(phrase['notes'])
            recent_notes = all_notes_so_far[-6:]
            notes_str = ", ".join([f"{note['pitch']}({note['beats']}b)" for note in recent_notes])
            context = f"Previous notes: {notes_str}"
        else:
            context = "This is the first phrase of the melody."

        # Create prompt
        base_prompt = f"""Generate a {emotion} {style} melody phrase at {bpm} BPM that outlines the chord {current_chord}.

Key: {self.key}
Current chord: {current_chord}
{context}

Create 6-12 musical notes for phrase {phrase_number + 1}.
Each note needs pitch and beats.
Make it sound {emotion}. Follow this melody description: {melody_description}.

Focus on 0.5 and 1.0 beat values for energy."""

        # Add chord targeting if enabled
        if use_targeting:
            chord_tones = self.get_chord_tones(current_chord)
            enhanced_prompt = base_prompt + f"""

CHORD GUIDELINES:
- Emphaseize these chord tones: {', '.join(chord_tones)}
- Start or end phrases on chord tones when possible
- Make the melody outline and emphasize {current_chord}
- The melody should sound harmonically connected to {current_chord}"""
        else:
            enhanced_prompt = base_prompt

        # Retry logic for empty phrases
        for attempt in range(max_retries):
            try:
                result = self.structured_model(
                    enhanced_prompt,
                    MelodyResponse,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature + (attempt * 0.1)
                )

                # Handle result
                if isinstance(result, str):
                    parsed = json.loads(result)
                    notes_data = parsed["notes"]
                    explanation = parsed.get("explanation", "")
                else:
                    notes_data = result.notes
                    explanation = result.explanation

                # Convert to dict format
                phrase_data = {
                    "notes": [
                        {"pitch": note.pitch, "beats": note.beats}
                        if hasattr(note, 'pitch') else note
                        for note in notes_data
                    ],
                    "chord": current_chord,
                    "explanation": explanation
                }

                # Check for empty phrases
                if len(phrase_data["notes"]) == 0:
                    print(f"   ⚠️  Attempt {attempt + 1}: Empty phrase, retrying...")
                    continue

                # Chord targeting analysis
                if use_targeting:
                    analysis = self.analyze_chord_targeting(phrase_data["notes"], current_chord)
                    phrase_data["targeting_analysis"] = analysis
                    targeting_info = f" | {analysis['analysis']}"
                else:
                    targeting_info = ""

                # Display results
                print(f"\nGenerated phrase {phrase_number + 1} over {current_chord} chord:")
                total_beats = 0
                for note in phrase_data["notes"]:
                    if use_targeting:
                        chord_tone_names = [n[:-1] for n in self.get_chord_tones(current_chord)]
                        is_chord_tone = note['pitch'][:-1] in chord_tone_names
                        indicator = "🎯" if is_chord_tone else "⚪"
                    else:
                        indicator = "♪"
                    print(f"  {indicator} {note['pitch']} {note['beats']}beats")
                    total_beats += note['beats']
                
                print(f"  Total: {total_beats}beats | Chord: {current_chord}{targeting_info}")
                
                if explanation:
                    print(f"  Explanation: {explanation}")

                self.running_melody.append(phrase_data)
                return phrase_data

            except Exception as e:
                print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"   ❌ All attempts failed, creating fallback")
                    fallback_phrase = self._create_fallback_phrase(current_chord)
                    self.running_melody.append(fallback_phrase)
                    return fallback_phrase

        return None

    def _create_fallback_phrase(self, chord_name: str) -> dict:
        """Create simple fallback phrase using chord tones."""
        chord_tones = self.get_chord_tones(chord_name)
        
        fallback_notes = [
            {"pitch": chord_tones[0], "beats": 1.0},
            {"pitch": chord_tones[1] if len(chord_tones) > 1 else chord_tones[0], "beats": 0.5},
            {"pitch": chord_tones[2] if len(chord_tones) > 2 else chord_tones[0], "beats": 1.0},
            {"pitch": chord_tones[0], "beats": 0.5}
        ]
        
        print(f"   🔧 Created fallback phrase with chord tones: {chord_tones}")
        
        return {
            "notes": fallback_notes,
            "chord": chord_name,
            "explanation": f"Fallback arpeggio using {chord_name} chord tones",
            "targeting_analysis": {"chord_tone_percentage": 100, "analysis": "100% chord tones (fallback)"}
        }

    def analyze_chord_targeting(self, phrase_notes: List[dict], chord_name: str) -> dict:
        """Analyze how well a phrase targets its chord."""
        chord_tone_names = [note[:-1] for note in self.get_chord_tones(chord_name)]
        
        total_notes = len([n for n in phrase_notes if n['pitch'] != 'rest'])
        if total_notes == 0:
            return {'chord_tone_percentage': 0, 'analysis': 'No notes generated'}
        
        chord_tone_count = 0
        for note in phrase_notes:
            if note['pitch'] != 'rest':
                note_name = note['pitch'][:-1]  # Remove octave
                if note_name in chord_tone_names:
                    chord_tone_count += 1
        
        percentage = (chord_tone_count / total_notes) * 100
        return {
            'chord_tone_percentage': percentage,
            'chord_tone_count': chord_tone_count,
            'total_notes': total_notes,
            'analysis': f"{percentage:.1f}% chord tones ({chord_tone_count}/{total_notes})"
        }

    def analyze_melody_targeting(self):
        """Analyze chord targeting for the entire melody."""
        if not self.running_melody:
            print("No melody to analyze")
            return

        print(f"\n🎯 MELODY CHORD TARGETING ANALYSIS")
        print("=" * 40)

        total_notes = 0
        total_chord_tones = 0

        for i, phrase in enumerate(self.running_melody, 1):
            if 'targeting_analysis' in phrase:
                analysis = phrase['targeting_analysis']
                print(f"Phrase {i} [{phrase['chord']}]: {analysis['analysis']}")
                total_notes += analysis.get('total_notes', len(phrase['notes']))
                total_chord_tones += analysis.get('chord_tone_count', 0)

        if total_notes > 0:
            overall_percentage = (total_chord_tones / total_notes) * 100
            print(f"\n🎼 OVERALL: {overall_percentage:.1f}% chord tones ({total_chord_tones}/{total_notes})")
            
            if overall_percentage >= 70:
                print("✅ Excellent chord targeting!")
            elif overall_percentage >= 50:
                print("⚠️  Moderate chord targeting")
            else:
                print("❌ Poor chord targeting")

        return {
            'total_notes': total_notes,
            'chord_tone_count': total_chord_tones,
            'percentage': overall_percentage if total_notes > 0 else 0
        }

    def get_total_beats(self):
        """Total composition length in beats."""
        return sum(
            sum(note['beats'] for note in phrase['notes'])
            for phrase in self.running_melody
        )

    def print_melody_with_chords(self):
        """Display melody with chord information."""
        if not self.orchestration:
            print("No melody generated.")
            return

        print(f"\n=== {self.orchestration['emotion'].title()} {self.orchestration['style'].title()} Melody ===")
        print(f"Key: {self.key} | BPM: {self.orchestration['bpm']} | Instrument: {self.orchestration['instrument']}")
        print(f"Chord Progression: {' → '.join(self.chord_progression)}")
        print()

        for i, phrase in enumerate(self.running_melody, 1):
            chord = phrase.get('chord', 'Unknown')
            notes = ", ".join([f"{n['pitch']}({n['beats']}b)" for n in phrase['notes']])
            phrase_beats = sum(n['beats'] for n in phrase['notes'])
            
            if 'targeting_analysis' in phrase:
                targeting = f" | {phrase['targeting_analysis']['analysis']}"
            else:
                targeting = ""
                
            print(f"Phrase {i} [{chord}]: {notes} ({phrase_beats}beats){targeting}")

        print(f"\nTotal: {self.get_total_beats()}beats")
