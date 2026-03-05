"""
Label mapping module for multi-dataset audio event detection.

Maps class labels from ESC-50, UrbanSound8K, FSD50K, and AudioSet
to our unified 50-class taxonomy.

Each mapping is: { source_dataset_label: our_label }
If a source label maps to None, it is skipped (no match in our taxonomy).
"""

# =============================================================================
# Our 50 target classes (grouped by category)
# =============================================================================

TARGET_CLASSES = [
    # Human vocal sounds (8)
    "speech", "laughter", "crying", "shouting", "whispering",
    "singing", "cough", "sneeze",
    # Human activity sounds (7)
    "footsteps", "running", "clapping", "cheering", "applause",
    "breathing", "snoring",
    # Household / indoor sounds (11)
    "door_knock", "door_open", "door_close", "glass_breaking",
    "dishes_clattering", "keyboard_typing", "mouse_click",
    "phone_ringing", "alarm_clock", "water_running", "toilet_flush",
    # Transportation sounds (8)
    "car_engine", "car_horn", "siren", "motorcycle", "train",
    "airplane", "helicopter", "engine_idling",
    # Animal sounds (5)
    "dog_bark", "cat_meow", "bird_chirping", "rooster_crow", "cow_moo",
    # Nature sounds (5)
    "rain", "thunder", "wind", "fire_crackling", "water_stream",
    # Mechanical / tool sounds (4)
    "hammering", "drilling", "saw_cutting", "machine_running",
    # Impact sounds (2)
    "object_drop", "explosion",
]

# =============================================================================
# ESC-50 label mapping
# ESC-50 has 50 classes, 5-second clips, 2000 total samples
# Source: https://github.com/karolpiczak/ESC-50
# =============================================================================

ESC50_MAPPING = {
    # Animals
    "dog": "dog_bark",
    "rooster": "rooster_crow",
    "pig": None,
    "cow": "cow_moo",
    "frog": None,
    "cat": "cat_meow",
    "hen": None,
    "insects": None,
    "sheep": None,
    "crow": None,
    # Natural soundscapes
    "rain": "rain",
    "sea_waves": None,
    "crackling_fire": "fire_crackling",
    "crickets": None,
    "chirping_birds": "bird_chirping",
    "water_drops": "water_stream",
    "wind": "wind",
    "pouring_water": "water_running",
    "toilet_flush": "toilet_flush",
    "thunderstorm": "thunder",
    # Human non-speech
    "crying_baby": "crying",
    "sneezing": "sneeze",
    "clapping": "clapping",
    "breathing": "breathing",
    "coughing": "cough",
    "footsteps": "footsteps",
    "laughing": "laughter",
    "brushing_teeth": None,
    "snoring": "snoring",
    "drinking_sipping": None,
    # Interior/domestic
    "door_wood_knock": "door_knock",
    "mouse_click": "mouse_click",
    "keyboard_typing": "keyboard_typing",
    "door_wood_creep": "door_open",
    "can_opening": None,
    "washing_machine": "machine_running",
    "vacuum_cleaner": "machine_running",
    "clock_alarm": "alarm_clock",
    "clock_tick": None,
    "glass_breaking": "glass_breaking",
    # Exterior/urban
    "helicopter": "helicopter",
    "chainsaw": "saw_cutting",
    "siren": "siren",
    "car_horn": "car_horn",
    "engine": "car_engine",
    "train": "train",
    "church_bells": None,
    "airplane": "airplane",
    "fireworks": "explosion",
    "hand_saw": "saw_cutting",
}

# =============================================================================
# UrbanSound8K label mapping
# 8732 clips, 10 classes, <=4 seconds each
# Source: https://urbansounddataset.weebly.com/urbansound8k.html
# =============================================================================

URBANSOUND8K_MAPPING = {
    "air_conditioner": "machine_running",
    "car_horn": "car_horn",
    "children_playing": "cheering",
    "dog_bark": "dog_bark",
    "drilling": "drilling",
    "engine_idling": "engine_idling",
    "gun_shot": "explosion",
    "jackhammer": "hammering",
    "siren": "siren",
    "street_music": None,
}

# Numeric class ID to name (UrbanSound8K uses numeric IDs in metadata)
URBANSOUND8K_CLASS_ID_TO_NAME = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

# =============================================================================
# FSD50K label mapping (AudioSet ontology labels)
# ~51k clips, 200 classes
# Source: https://zenodo.org/record/4060432
#
# FSD50K uses AudioSet ontology labels. We map the relevant ones.
# Many FSD50K clips have multiple labels — all matching labels are kept.
# =============================================================================

FSD50K_MAPPING = {
    # Human vocal sounds
    "Speech": "speech",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Child speech, kid speaking": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Laughter": "laughter",
    "Crying, sobbing": "crying",
    "Baby cry, infant cry": "crying",
    "Shout": "shouting",
    "Screaming": "shouting",
    "Yell": "shouting",
    "Whispering": "whispering",
    "Singing": "singing",
    "Choir": "singing",
    "Cough": "cough",
    "Sneeze": "sneeze",
    # Human activity sounds
    "Walk, footsteps": "footsteps",
    "Footsteps": "footsteps",
    "Run": "running",
    "Running": "running",
    "Clapping": "clapping",
    "Hands": "clapping",
    "Cheering": "cheering",
    "Applause": "applause",
    "Crowd": "cheering",
    "Breathing": "breathing",
    "Snoring": "snoring",
    # Household / indoor sounds
    "Knock": "door_knock",
    "Door": "door_close",
    "Doorbell": "door_knock",
    "Slam": "door_close",
    "Glass": "glass_breaking",
    "Shatter": "glass_breaking",
    "Breaking": "glass_breaking",
    "Dishes, pots, and pans": "dishes_clattering",
    "Cutlery, silverware": "dishes_clattering",
    "Computer keyboard": "keyboard_typing",
    "Typing": "keyboard_typing",
    "Clickety-clack": "keyboard_typing",
    "Mouse": "mouse_click",
    "Click": "mouse_click",
    "Telephone": "phone_ringing",
    "Telephone bell ringing": "phone_ringing",
    "Ringtone": "phone_ringing",
    "Alarm clock": "alarm_clock",
    "Alarm": "alarm_clock",
    "Water tap, faucet": "water_running",
    "Sink (filling or washing)": "water_running",
    "Fill (with liquid)": "water_running",
    "Bathtub (filling or washing)": "water_running",
    "Toilet flush": "toilet_flush",
    # Transportation sounds
    "Car": "car_engine",
    "Vehicle": "car_engine",
    "Car passing by": "car_engine",
    "Vehicle horn, car horn, honking": "car_horn",
    "Honk": "car_horn",
    "Air horn, truck horn": "car_horn",
    "Siren": "siren",
    "Civil defense siren": "siren",
    "Ambulance (siren)": "siren",
    "Fire engine, fire truck (siren)": "siren",
    "Police car (siren)": "siren",
    "Motorcycle": "motorcycle",
    "Train": "train",
    "Train horn": "train",
    "Railroad car, train wagon": "train",
    "Rail transport": "train",
    "Fixed-wing aircraft, airplane": "airplane",
    "Aircraft": "airplane",
    "Jet engine": "airplane",
    "Propeller, airscrew": "airplane",
    "Helicopter": "helicopter",
    "Engine": "engine_idling",
    "Engine starting": "engine_idling",
    "Idling": "engine_idling",
    "Light engine (high frequency)": "engine_idling",
    "Medium engine (mid frequency)": "engine_idling",
    "Heavy engine (low frequency)": "engine_idling",
    # Animal sounds
    "Dog": "dog_bark",
    "Bark": "dog_bark",
    "Growling": "dog_bark",
    "Bow-wow": "dog_bark",
    "Cat": "cat_meow",
    "Meow": "cat_meow",
    "Purr": "cat_meow",
    "Hiss": "cat_meow",
    "Bird": "bird_chirping",
    "Bird vocalization, bird call, bird song": "bird_chirping",
    "Chirp, tweet": "bird_chirping",
    "Pigeon, dove": "bird_chirping",
    "Crow": "bird_chirping",
    "Owl": "bird_chirping",
    "Rooster, cock-a-doodle-doo": "rooster_crow",
    "Chicken, rooster": "rooster_crow",
    "Cattle, bovinae": "cow_moo",
    "Moo": "cow_moo",
    # Nature sounds
    "Rain": "rain",
    "Rain on surface": "rain",
    "Raindrop": "rain",
    "Thunder": "thunder",
    "Thunderstorm": "thunder",
    "Wind": "wind",
    "Wind noise (microphone)": None,  # not actual wind
    "Howl": None,
    "Fire": "fire_crackling",
    "Crackle": "fire_crackling",
    "Fireplace, wood-burning fireplace": "fire_crackling",
    "Stream": "water_stream",
    "Waterfall": "water_stream",
    "Brook": "water_stream",
    # Mechanical / tool sounds
    "Hammer": "hammering",
    "Jackhammer": "hammering",
    "Drill": "drilling",
    "Power tool": "drilling",
    "Saw": "saw_cutting",
    "Chainsaw": "saw_cutting",
    "Sawing": "saw_cutting",
    "Mechanical fan": "machine_running",
    "Sewing machine": "machine_running",
    "Engine knocking": "machine_running",
    "Printer": "machine_running",
    # Impact sounds
    "Thump, thud": "object_drop",
    "Bang": "object_drop",
    "Smash, crash": "object_drop",
    "Bouncing": "object_drop",
    "Explosion": "explosion",
    "Burst, pop": "explosion",
    "Gunshot, gunfire": "explosion",
    "Boom": "explosion",
    "Fireworks": "explosion",
}

# =============================================================================
# AudioSet label mapping (for optional AudioSet download)
# Uses AudioSet mid-level ontology IDs → our classes
# =============================================================================

AUDIOSET_MID_MAPPING = {
    # Human vocal
    "/m/09x0r": "speech",           # Speech
    "/m/015lz1": "singing",         # Singing
    "/m/01j3sz": "laughter",        # Laughter
    "/m/07r660_": "crying",         # Crying, sobbing
    "/m/01d3sd": "shouting",        # Shout
    "/m/02rtxlg": "whispering",     # Whispering
    "/m/01b_21": "cough",           # Cough
    "/m/0dl9sf8": "sneeze",         # Sneeze
    # Human activity
    "/m/07pbtc8": "footsteps",      # Walk, footsteps
    "/m/06h7j": "running",          # Run
    "/m/07rkbfh": "clapping",       # Clapping
    "/m/03qtwd": "cheering",        # Cheering
    "/m/0lyf6": "applause",         # Applause
    "/m/01d3_f": "breathing",       # Breathing
    "/m/01hsr_": "snoring",         # Snoring
    # Household
    "/m/07q6cd_": "door_knock",     # Knock
    "/m/02dgv": "door_close",       # Door
    "/m/07q0yl5": "glass_breaking", # Shatter
    "/m/04brg2": "dishes_clattering", # Dishes
    "/m/01m2v": "keyboard_typing",  # Computer keyboard
    "/m/0dxrf": "phone_ringing",    # Telephone bell ringing
    "/m/046dlr": "alarm_clock",     # Alarm clock
    "/m/05kq4": "water_running",    # Water tap, faucet
    "/m/03w41f": "toilet_flush",    # Toilet flush
    # Transportation
    "/m/012f08": "car_engine",      # Car
    "/m/0148z0": "car_horn",        # Vehicle horn
    "/m/03kmc9": "siren",           # Siren
    "/m/04_sv": "motorcycle",       # Motorcycle
    "/m/07jdr": "train",            # Train
    "/m/0cmf2": "airplane",         # Aircraft
    "/m/09ct_": "helicopter",       # Helicopter
    "/m/01j4z9": "engine_idling",   # Engine
    # Animal
    "/m/05tny_": "dog_bark",        # Bark
    "/m/02yds9": "cat_meow",        # Meow
    "/m/015p6": "bird_chirping",    # Bird
    "/m/07st89h": "rooster_crow",   # Rooster
    "/m/07bgp": "cow_moo",          # Cattle, bovinae
    # Nature
    "/m/0838f": "rain",             # Rain
    "/m/0ngt1": "thunder",          # Thunder
    "/m/0t49": "wind",              # Wind
    "/m/04k94": "fire_crackling",   # Fire
    "/m/0j6m2": "water_stream",     # Stream
    # Mechanical
    "/m/03l9g": "hammering",        # Hammer
    "/m/01b82r": "drilling",        # Drill
    "/m/02p01q": "saw_cutting",     # Chainsaw
    "/m/01lsmm": "machine_running", # Mechanical fan
    # Impact
    "/m/02_nn": "explosion",        # Explosion
}


def get_mapping(dataset_name: str) -> dict:
    """
    Get the label mapping for a given dataset.

    Args:
        dataset_name: One of 'esc50', 'urbansound8k', 'fsd50k', 'audioset'.

    Returns:
        Dict mapping source labels to our target labels (or None to skip).
    """
    mappings = {
        "esc50": ESC50_MAPPING,
        "urbansound8k": URBANSOUND8K_MAPPING,
        "fsd50k": FSD50K_MAPPING,
        "audioset": AUDIOSET_MID_MAPPING,
    }
    name = dataset_name.lower().replace("-", "").replace("_", "")
    for key, mapping in mappings.items():
        if key.replace("_", "") == name:
            return mapping
    raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(mappings.keys())}")


def map_labels(source_labels: list, mapping: dict) -> list:
    """
    Map a list of source labels to our target taxonomy.

    Args:
        source_labels: List of labels from the source dataset.
        mapping: The label mapping dict.

    Returns:
        List of unique target labels (empty if no matches).
    """
    target = set()
    for label in source_labels:
        label_str = str(label).strip()
        if label_str in mapping and mapping[label_str] is not None:
            target.add(mapping[label_str])
    return sorted(target)


def get_coverage_report(mapping: dict) -> dict:
    """
    Show which of our 50 target classes are covered by a mapping.

    Returns:
        Dict with 'covered', 'missing', 'coverage_pct' keys.
    """
    covered = set(v for v in mapping.values() if v is not None)
    missing = set(TARGET_CLASSES) - covered
    return {
        "covered": sorted(covered),
        "missing": sorted(missing),
        "covered_count": len(covered),
        "total": len(TARGET_CLASSES),
        "coverage_pct": len(covered) / len(TARGET_CLASSES) * 100,
    }
