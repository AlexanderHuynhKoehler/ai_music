
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# ---------- Minimal Schema ----------
def make_minimal_schema(max_notes: int = 8) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "bpm": {"type": "integer", "minimum": 60, "maximum": 200},
            "notes": {
                "type": "array",
                "maxItems": max_notes,
                "items": {
                    "type": "object",
                    "properties": {
                        "pitch": {"type": "string"},
                        "start": {"type": "number", "minimum": 0},
                        "duration": {"type": "number", "minimum": 0.25}
                    },
                    "required": ["pitch", "start", "duration"]
                }
            }
        },
        "required": ["bpm", "notes"]
    }

# ---------- Prompt ----------
def build_simple_prompt(key: str, bpm: int, num_notes: int) -> str:
    return f"""Generate a melody as JSON. Key: {key}, BPM: {bpm}, {num_notes} notes.

Format:
{{"bpm": {bpm}, "notes": [{{"pitch": "D4", "start": 0, "duration": 1}}, {{"pitch": "E4", "start": 1, "duration": 0.5}}]}}

Rules:
- Pitch: Scientific notation (C4, F#3, Bb5)
- Start: Beat position (0, 0.5, 1, 1.5...)
- Duration: Note length in beats (0.25, 0.5, 1, 2...)
- Stay in {key} scale mostly
- No overlapping notes

JSON only:"""

# ---------- Repair JSON ----------
def repair_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end+1]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return json.loads(text)

# ---------- Pitch name → MIDI ----------
def add_midi_pitches(melody_json: dict) -> dict:
    note_base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
    accidental = {"#":1,"b":-1}

    def name_to_midi(name: str) -> int:
        letter = name[0].upper()
        acc = 0
        idx = 1
        if len(name) > 2 and name[1] in accidental:
            acc = accidental[name[1]]
            idx = 2
        octave = int(name[idx:])
        return 12 + note_base[letter] + acc + 12*octave

    for note in melody_json.get("notes", []):
        if note["pitch"].lower() not in ("rest", "r", ""):
            note["midi"] = name_to_midi(note["pitch"])
        else:
            note["midi"] = None
    return melody_json

# ---------- Generation ----------
def generate_melody_with_midi(
    pipe,
    *,
    key: str = "D major",
    bpm: int = 90,
    num_notes: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    prompt = build_simple_prompt(key, bpm, num_notes)
    input_ids = pipe.tokenizer.encode(prompt, return_tensors="pt").to(pipe.model.device)

    schema = make_minimal_schema(max_notes=num_notes + 2)
    parser = JsonSchemaParser(schema)
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        pipe.tokenizer, parser
    )

    try:
        with torch.no_grad():
            out = pipe.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:]
        text = pipe.tokenizer.decode(gen_ids, skip_special_tokens=True)
        result = json.loads(text)
    except Exception:
        with torch.no_grad():
            out = pipe.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
            )
        text = pipe.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        result = repair_json(text)

    return add_midi_pitches(result)

# ---------- Setup ----------
def setup_llama_pipeline(model_name: str = MODEL_NAME, login_token: str | None = None):
    if login_token or os.getenv("HUGGINGFACE_TOKEN"):
        hf_login(token=login_token or os.getenv("HUGGINGFACE_TOKEN"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    class Pipeline:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
    return Pipeline(model, tokenizer)

  pipe = setup_llama_pipeline()
  melody = generate_melody_with_midi(pipe, key="G major", bpm=120, num_notes=6, temperature=0.8)
  print(json.dumps(melody, indent=2))
