class MultiLLMDynamicsGenerator:
    def __init__(self, pipe):
        """Initialize multi-LLM dynamics generator."""
        self.model = outlines.from_transformers(pipe.model, pipe.tokenizer)
    
    def generate_dynamics(self, running_melody: List[dict], orchestration: dict) -> List[dict]:
        """Generate dynamics using 2-LLM approach for reliability."""
        
        # Flatten all notes from all phrases
        all_notes = []
        for phrase in running_melody:
            all_notes.extend(phrase['notes'])

        print(f"Generating dynamics for {len(all_notes)} notes using multi-LLM...")
        
        # Create concise notes list
        notes_str = ", ".join([f"{note['pitch']}({note['beats']}b)" for note in all_notes])
        emotion, style = orchestration['emotion'], orchestration['style']
        
        # LLM Task 1: Generate velocities and articulations
        velocities, articulations = self._generate_expression(notes_str, len(all_notes), emotion, style)
        
        # LLM Task 2: Generate timing parameters
        start_offsets, tie_nexts = self._generate_timing(notes_str, len(all_notes), emotion, style)
        
        # Recombine into complete events
        complete_events = []
        for i, note in enumerate(all_notes):
            complete_events.append({
                'pitch': note['pitch'],
                'beats': note['beats'],
                'bpm': orchestration['bpm'],
                'velocity': velocities[i],
                'articulation': articulations[i],
                'start_offset_beats': start_offsets[i],
                'tie_next': tie_nexts[i]
            })
        
        print(f"✅ Generated {len(complete_events)} complete events with dynamics")
        return complete_events

    def _generate_expression(self, notes_str: str, count: int, emotion: str, style: str):
        """LLM 1: Generate velocities and articulations."""
        
        prompt = f"""Assign velocity (40-127) and articulation to {count} {emotion} {style} notes.

Notes: {notes_str}

Guidelines:
- Angry: velocity 100-127, use "accent" or "staccato"
- Sad: velocity 40-70, use "legato" 
- Happy: velocity 70-100, use "staccato"
- Jazz: velocity 60-90, mix "legato" and "accent"

OUTPUT FORMAT: {{"velocities": [85, 110, 95], "articulations": ["accent", "staccato", null]}}

Generate {count} velocities and {count} articulations for {emotion} {style}."""

        try:
            result = self.model(
                prompt, VelocityArticulationResponse, 
                max_new_tokens=800, temperature=0.1
            )
            
            if isinstance(result, str):
                data = json.loads(result)
                velocities, articulations = data['velocities'], data['articulations']
            else:
                velocities, articulations = result.velocities, result.articulations
            
            # Ensure correct length
            velocities = self._fix_length(velocities, count, 80)
            articulations = self._fix_length(articulations, count, None)
            
            print(f"  ✅ Expression: {len(velocities)} velocities, {len(articulations)} articulations")
            return velocities, articulations
            
        except Exception as e:
            print(f"  ❌ Expression failed: {e}")
            return self._get_expression_fallback(count, emotion)

    def _generate_timing(self, notes_str: str, count: int, emotion: str, style: str):
        """LLM 2: Generate timing parameters."""
        
        prompt = f"""Assign timing to {count} {emotion} {style} notes.

Notes: {notes_str}

OUTPUT: {{"start_offsets": [0.0], "tie_nexts": [false]}}"""

        try:
            result = self.model(
                prompt, TimingResponse, 
                max_new_tokens=300, temperature=0.1
            )
            
            if isinstance(result, str):
                data = json.loads(result)
                start_offsets, tie_nexts = data['start_offsets'], data['tie_nexts']
            else:
                start_offsets, tie_nexts = result.start_offsets, result.tie_nexts
            
            # Ensure correct length
            start_offsets = self._fix_length(start_offsets, count, 0.0)
            tie_nexts = self._fix_length(tie_nexts, count, False)
            
            print(f"  ✅ Timing: {len(start_offsets)} offsets, {len(tie_nexts)} ties")
            return start_offsets, tie_nexts
            
        except Exception as e:
            print(f"  ❌ Timing failed: {e}")
            return self._get_timing_fallback(count)

    def _fix_length(self, lst: List, target: int, default):
        """Ensure list is exactly target length."""
        if len(lst) == target:
            return lst
        elif len(lst) > target:
            return lst[:target]
        else:
            return lst + [default] * (target - len(lst))

    def _get_expression_fallback(self, count: int, emotion: str):
        """Fallback for expression generation."""
        base_vel = 110 if emotion == 'angry' else 50 if emotion == 'sad' else 80
        default_art = 'accent' if emotion == 'angry' else 'legato' if emotion == 'sad' else None
        
        velocities = [base_vel + (i % 15) for i in range(count)]
        articulations = [default_art] * count
        
        return velocities, articulations

    def _get_timing_fallback(self, count: int):
        """Fallback for timing generation."""
        return [0.0] * count, [False] * count
