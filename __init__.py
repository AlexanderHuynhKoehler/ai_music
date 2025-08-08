# Expose key functions and classes at the package level for easy imports

from .generator import (
    make_minimal_schema,
    build_simple_prompt,
    repair_json,
    generate_melody_json_improved,
    generate_melody_with_midi,
    setup_llama_pipeline,
)

from .events import Event, InstrumentTrack
from .utils import note_name_to_midi, midi_to_note_name

__all__ = [
    # Generator functions
    "make_minimal_schema",
    "build_simple_prompt",
    "repair_json",
    "generate_melody_json_improved",
    "generate_melody_with_midi",
    "setup_llama_pipeline",

    # Event system
    "Event",
    "InstrumentTrack",

    # Utilities
    "note_name_to_midi",
    "midi_to_note_name",
]
