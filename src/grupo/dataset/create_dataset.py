"""
Generate synthetic Spotify tracks dataset with realistic audio features.
100,000 tracks across 20 genres with per-genre feature distributions.

Popularity is modelled as a Zipf-like base with small contributions from
release year, genre identity and audio features.  The signal is deliberately
kept weak so that R² in a regression model stays in the ~0.04-0.08 range,
confirming that audio features alone cannot reliably predict popularity.

Intentional data quality issues are injected into audio feature columns
to simulate a real-world dirty dataset for the Data Preparation phase:
  - Missing values  (~3 % per audio feature column)
  - Numeric outliers (~0.7 % per affected column)
  - Duplicate rows     (~0.5 % of total rows)

A ground-truth log is exported to spotify_tracks_errors_log.csv.
"""

import numpy as np
import pandas as pd
import random
import string
from pathlib import Path

np.random.seed(42)
random.seed(42)

N = 100_000
OUTPUT       = Path(__file__).parent / "spotify_tracks.csv"
OUTPUT_ERRORS = Path(__file__).parent / "spotify_tracks_errors_log.csv"

# Genre definitions with characteristic audio feature means
GENRES = {
    "pop":              dict(dance=0.72, energy=0.70, acoustic=0.15, speech=0.06, valence=0.65, tempo=118, loud=-5.5, instr=0.01),
    "hip-hop":          dict(dance=0.82, energy=0.68, acoustic=0.10, speech=0.22, valence=0.55, tempo=95,  loud=-6.0, instr=0.01),
    "rock":             dict(dance=0.50, energy=0.82, acoustic=0.12, speech=0.05, valence=0.50, tempo=130, loud=-5.0, instr=0.05),
    "metal":            dict(dance=0.35, energy=0.93, acoustic=0.05, speech=0.06, valence=0.35, tempo=148, loud=-4.0, instr=0.12),
    "electronic":       dict(dance=0.78, energy=0.85, acoustic=0.05, speech=0.05, valence=0.60, tempo=128, loud=-5.5, instr=0.25),
    "classical":        dict(dance=0.25, energy=0.25, acoustic=0.88, speech=0.04, valence=0.45, tempo=108, loud=-15., instr=0.90),
    "jazz":             dict(dance=0.55, energy=0.40, acoustic=0.65, speech=0.05, valence=0.60, tempo=120, loud=-12., instr=0.55),
    "r&b":              dict(dance=0.75, energy=0.62, acoustic=0.22, speech=0.08, valence=0.60, tempo=100, loud=-7.0, instr=0.02),
    "country":          dict(dance=0.60, energy=0.62, acoustic=0.48, speech=0.04, valence=0.68, tempo=116, loud=-7.5, instr=0.02),
    "folk":             dict(dance=0.45, energy=0.38, acoustic=0.82, speech=0.04, valence=0.55, tempo=112, loud=-12., instr=0.10),
    "reggae":           dict(dance=0.72, energy=0.58, acoustic=0.30, speech=0.07, valence=0.78, tempo=90,  loud=-9.0, instr=0.05),
    "latin":            dict(dance=0.82, energy=0.75, acoustic=0.18, speech=0.07, valence=0.80, tempo=110, loud=-6.5, instr=0.02),
    "indie":            dict(dance=0.55, energy=0.60, acoustic=0.38, speech=0.05, valence=0.52, tempo=122, loud=-8.0, instr=0.08),
    "blues":            dict(dance=0.52, energy=0.50, acoustic=0.55, speech=0.05, valence=0.55, tempo=105, loud=-11., instr=0.08),
    "soul":             dict(dance=0.68, energy=0.58, acoustic=0.35, speech=0.06, valence=0.65, tempo=98,  loud=-8.5, instr=0.04),
    "punk":             dict(dance=0.45, energy=0.90, acoustic=0.06, speech=0.08, valence=0.45, tempo=168, loud=-5.0, instr=0.03),
    "drum-and-bass":    dict(dance=0.75, energy=0.90, acoustic=0.03, speech=0.04, valence=0.50, tempo=174, loud=-6.0, instr=0.35),
    "ambient":          dict(dance=0.25, energy=0.18, acoustic=0.55, speech=0.03, valence=0.35, tempo=80,  loud=-18., instr=0.75),
    "gospel":           dict(dance=0.60, energy=0.65, acoustic=0.42, speech=0.07, valence=0.82, tempo=108, loud=-8.0, instr=0.02),
    "k-pop":            dict(dance=0.78, energy=0.78, acoustic=0.12, speech=0.07, valence=0.70, tempo=124, loud=-5.5, instr=0.02),
}
GENRE_LIST = list(GENRES.keys())

# Small per-genre popularity bias (~6 % of total variance)
# Genres with mainstream appeal get a slight boost
GENRE_POP_BIAS = {
    "pop":              6,
    "hip-hop":          8,
    "rock":             3,
    "metal":           -2,
    "electronic":       5,
    "classical":       -3,
    "jazz":            -1,
    "r&b":              4,
    "country":          1,
    "folk":            -2,
    "reggae":           2,
    "latin":            4,
    "indie":           -1,
    "blues":           -2,
    "soul":             1,
    "punk":             0,
    "drum-and-bass":    3,
    "ambient":         -4,
    "gospel":           1,
    "k-pop":            7,
}

# Weighted genre distribution (pop/hip-hop/rock dominate)
_gw = np.array([0.15, 0.13, 0.12, 0.05, 0.08, 0.05, 0.05, 0.07, 0.06, 0.04,
                0.03, 0.04, 0.04, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.05])
GENRE_WEIGHTS = _gw / _gw.sum()

KEYS = list(range(12))
KEY_WEIGHTS = [0.12, 0.06, 0.10, 0.05, 0.09, 0.09, 0.06, 0.11, 0.06, 0.09, 0.05, 0.12]

FIRST_WORDS = ["The", "My", "Your", "One", "Night", "Lost", "Blue", "Dark",
               "Golden", "Broken", "Wild", "Slow", "Fast", "Last", "Forever",
               "Electric", "Silent", "Burning", "Rising", "Falling"]
SECOND_WORDS = ["Dream", "Heart", "Fire", "Sky", "Light", "Wave", "Road",
                "Soul", "Rain", "Storm", "Star", "Dance", "Song", "Love",
                "Night", "Day", "World", "Moon", "River", "Mountain"]
SUFFIXES = ["", "", "", " (feat. someone)", " (Remix)", " (Acoustic Version)",
            " - Radio Edit", " (Live)", ""]

ARTISTS_FIRST = ["Alex", "Jordan", "Sam", "Chris", "Taylor", "Morgan", "Riley",
                 "Drew", "Casey", "Avery", "Blake", "Quinn", "Skyler", "Jamie",
                 "Logan", "Cameron", "Reese", "Dana", "Jesse", "Devon"]
ARTISTS_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
                "Wilson", "Anderson", "Thomas", "Jackson", "White", "Harris",
                "Martin", "Thompson"]
BAND_NAMES = ["The Midnight", "Neon Pulse", "Echo Chamber", "Static Wave",
              "Velvet Underground", "Crystal Veil", "Iron Soul", "The Drifters",
              "Phantom Keys", "Solar Drift", "Lunar Echo", "Storm Collective",
              "Wild Current", "The Architects", "Desert Wind"]


def rand_track_id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=22))


def rand_artist():
    if random.random() < 0.3:
        return random.choice(BAND_NAMES)
    return f"{random.choice(ARTISTS_FIRST)} {random.choice(ARTISTS_LAST)}"


def rand_track_name():
    name = f"{random.choice(FIRST_WORDS)} {random.choice(SECOND_WORDS)}"
    return name + random.choice(SUFFIXES)


def rand_album(artist, year):
    styles = [
        f"{random.choice(FIRST_WORDS)} {random.choice(SECOND_WORDS)}",
        f"{artist.split()[0]}'s Greatest Hits",
        f"Volume {random.randint(1, 3)}",
        f"{year} Sessions",
        f"The {random.choice(SECOND_WORDS)} Album",
    ]
    return random.choice(styles)


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def generate_features(genre_params):
    p = genre_params
    return dict(
        danceability=clip01(np.random.normal(p["dance"], 0.12)),
        energy=clip01(np.random.normal(p["energy"], 0.12)),
        loudness=float(np.clip(np.random.normal(p["loud"], 3.0), -60, 0)),
        speechiness=clip01(np.random.normal(p["speech"], 0.04)),
        acousticness=clip01(np.random.normal(p["acoustic"], 0.12)),
        instrumentalness=clip01(np.random.normal(p["instr"], 0.08)),
        liveness=clip01(np.random.normal(0.18, 0.10)),
        valence=clip01(np.random.normal(p["valence"], 0.15)),
        tempo=float(np.clip(np.random.normal(p["tempo"], 15), 50, 220)),
    )


rows = []
for i in range(N):
    genre = np.random.choice(GENRE_LIST, p=GENRE_WEIGHTS)
    gp = GENRES[genre]
    _yw = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                    1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                    3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 3.5])
    year = int(np.random.choice(range(2000, 2025), p=_yw / _yw.sum()))

    artist = rand_artist()
    feats = generate_features(gp)

    # Popularity: Zipf-like base distribution
    # Small contributions from genre, year and audio features
    # The effect is intentionally weak → R² remains low
    base_pop = int(np.clip(np.random.exponential(18), 0, 90))
    year_bonus = int((year - 2000) / 24 * 8)
    genre_bias = GENRE_POP_BIAS[genre]

    # Subtle audio-feature signal (~±4 pts max combined)
    feature_bonus = (
        2.5 * (feats["danceability"] - gp["dance"]) +
        1.5 * (feats["energy"]       - gp["energy"]) +
        2.0 * (feats["valence"]      - gp["valence"]) -
        2.0 * (feats["acousticness"] - gp["acoustic"]) +
        1.0 * (feats["speechiness"]  - gp["speech"])
    )

    popularity = int(min(100, max(0,
        base_pop + year_bonus + genre_bias + feature_bonus + random.randint(-10, 10)
    )))

    duration_ms = int(np.clip(np.random.normal(210_000, 45_000), 90_000, 600_000))
    explicit = random.random() < (0.25 if genre in ["hip-hop", "metal", "punk", "r&b"] else 0.08)
    key = int(np.random.choice(KEYS, p=KEY_WEIGHTS))
    mode = 1 if random.random() < 0.62 else 0
    time_sig = int(np.random.choice([3, 4, 5, 6, 7], p=[0.06, 0.88, 0.03, 0.02, 0.01]))

    rows.append({
        "track_id": rand_track_id(),
        "track_name": rand_track_name(),
        "artist_name": artist,
        "album_name": rand_album(artist, year),
        "release_year": year,
        "genre": genre,
        "popularity": popularity,
        "duration_ms": duration_ms,
        "explicit": explicit,
        **feats,
        "key": key,
        "mode": mode,
        "time_signature": time_sig,
    })

df = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# ERROR INJECTION
# Uses a separate RNG so the clean data generation above stays reproducible.
# ──────────────────────────────────────────────────────────────────────────────

_err_rng = np.random.default_rng(99)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

OUTLIER_COLS = {
    "tempo":      dict(high=(300, 420), low=(0, 15)),
    "loudness":   dict(high=(1, 10),   low=(-90, -70)),
    "danceability": dict(high=(1.05, 1.5), low=(-0.3, -0.01)),
    "energy":       dict(high=(1.05, 1.5), low=(-0.3, -0.01)),
    "valence":      dict(high=(1.05, 1.5), low=(-0.3, -0.01)),
}

MISSING_RATE  = 0.030
OUTLIER_RATE  = 0.007
DUPLICATE_N   = 500   # absolute number of duplicate rows added

error_log = []  # list of dicts → ground-truth CSV


def _available_mask(df, col):
    """Boolean mask: cells not already NaN and not already a string (format error)."""
    s = df[col]
    not_na = s.notna()
    not_str = s.apply(lambda v: not isinstance(v, str))
    return not_na & not_str


# 1. Missing values ────────────────────────────────────────────────────────────
for col in AUDIO_FEATURES:
    n_missing = int(len(df) * MISSING_RATE)
    idx = _err_rng.choice(df.index, size=n_missing, replace=False)
    df.loc[idx, col] = np.nan
    for i in idx:
        error_log.append({"row_index": int(i), "column": col,
                           "error_type": "missing_value", "detail": ""})

# 2. Numeric outliers ──────────────────────────────────────────────────────────
for col, bounds in OUTLIER_COLS.items():
    avail = df.index[_available_mask(df, col)].tolist()
    n_out = int(len(df) * OUTLIER_RATE)
    # split evenly between high and low
    n_high = n_out // 2
    n_low  = n_out - n_high
    chosen = _err_rng.choice(avail, size=n_high + n_low, replace=False)
    high_idx, low_idx = chosen[:n_high], chosen[n_high:]

    lo_h, hi_h = bounds["high"]
    for i in high_idx:
        val = float(_err_rng.uniform(lo_h, hi_h))
        df.loc[i, col] = val
        error_log.append({"row_index": int(i), "column": col,
                           "error_type": "outlier_high", "detail": f"{val:.4f}"})

    lo_l, hi_l = bounds["low"]
    for i in low_idx:
        val = float(_err_rng.uniform(lo_l, hi_l))
        df.loc[i, col] = val
        error_log.append({"row_index": int(i), "column": col,
                           "error_type": "outlier_low", "detail": f"{val:.4f}"})

# 4. Duplicate rows ────────────────────────────────────────────────────────────
dup_source_idx = _err_rng.choice(df.index, size=DUPLICATE_N, replace=False)
dup_rows = df.loc[dup_source_idx].copy()
df = pd.concat([df, dup_rows], ignore_index=True)

new_start = len(df) - DUPLICATE_N
for offset, orig_idx in enumerate(dup_source_idx):
    new_idx = new_start + offset
    error_log.append({"row_index": new_idx, "column": "",
                       "error_type": "duplicate",
                       "detail": f"copy of original row {int(orig_idx)}"})

# ──────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)

errors_df = pd.DataFrame(error_log,
                          columns=["row_index", "column", "error_type", "detail"])
errors_df.sort_values("row_index", inplace=True)
errors_df.to_csv(OUTPUT_ERRORS, index=False)

# Summary
print(f"Saved {len(df):,} rows to {OUTPUT}")
print(f"Saved {len(errors_df):,} error entries to {OUTPUT_ERRORS}\n")

summary = errors_df.groupby("error_type").size().rename("count")
print("── Error summary ──────────────────────────────")
print(summary.to_string())
print("───────────────────────────────────────────────")
print(f"\nTotal clean rows  : {N:,}")
print(f"Duplicate rows added: {DUPLICATE_N:,}")
print(f"Total rows in CSV : {len(df):,}")

