"""Column schemas of the shared tables and value parsers shared by all builders.

Conventions
- Every table is row-aligned with ``users.parquet.idx`` (dense 0..N-1).
- Nulls are preserved: a JSON null / absent field / TwiBot-20 ``"None "`` becomes null, never 0 or
  ``''``. An empty string stays ``''``. Fields a dataset does not have at all are all-null columns
  and are listed in ``profile.meta.json["absent_columns"]``.
- ``raw_*`` columns keep the exact release spelling (TwiBot-20 trailing spaces included) for code
  that compares raw strings.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa

SPLITS = ("train", "val", "test", "support")
SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}

USERS_SCHEMA = pa.schema([
    ("idx", pa.int64()),
    ("user_id", pa.string()),          # decimal digits
    ("raw_id", pa.string()),           # exactly as in the release (TwiBot-22: 'u123')
    ("split", pa.dictionary(pa.int8(), pa.string())),  # train / val / test / support
    ("label", pa.int8()),              # 1 bot, 0 human, -1 unlabelled
    ("raw_order", pa.int64()),         # position in the raw user file (TwiBot-20: within source file)
    ("split_csv_order", pa.int64()),   # TwiBot-22 row in split.csv, else -1
    ("label_csv_order", pa.int64()),   # TwiBot-22 row in label.csv, else -1
    ("source_file", pa.dictionary(pa.int8(), pa.string())),
])

STR = pa.string()
I64 = pa.int64()
BOOL = pa.bool_()
TS = pa.timestamp("s", tz="UTC")

PROFILE_FIELDS: list[tuple[str, pa.DataType]] = [
    ("idx", I64),
    ("profile_missing", BOOL),   # True when the whole profile object is null/absent
    ("name", STR), ("screen_name", STR), ("description", STR), ("url", STR), ("location", STR),
    ("profile_image_url", STR), ("created_at", TS),
    ("followers_count", I64), ("following_count", I64), ("statuses_count", I64),
    ("listed_count", I64), ("favourites_count", I64),
    ("protected", BOOL), ("verified", BOOL), ("default_profile", BOOL),
    ("default_profile_image", BOOL), ("geo_enabled", BOOL), ("profile_use_background_image", BOOL),
    ("profile_background_tile", BOOL), ("has_extended_profile", BOOL),
    ("contributors_enabled", BOOL), ("is_translator", BOOL), ("is_translation_enabled", BOOL),
    ("lang", STR), ("utc_offset", I64), ("time_zone", STR),
    ("pinned_tweet_id", STR), ("withheld", STR), ("entities", STR),
    # exact raw spellings
    ("raw_name", STR), ("raw_screen_name", STR), ("raw_description", STR), ("raw_location", STR),
    ("raw_profile_image_url", STR), ("raw_default_profile_image", STR), ("raw_url", STR),
    ("raw_protected", STR), ("raw_verified", STR), ("raw_created_at", STR),
]
PROFILE_SCHEMA = pa.schema(PROFILE_FIELDS)
PROFILE_COLUMNS = [n for n, _ in PROFILE_FIELDS]

COUNT_COLUMNS = ["followers_count", "following_count", "statuses_count", "listed_count",
                 "favourites_count", "utc_offset"]
BOOL_COLUMNS = [n for n, t in PROFILE_FIELDS if t == BOOL and n != "profile_missing"]


def empty_profile_row(idx: int) -> dict:
    row = {n: None for n in PROFILE_COLUMNS}
    row["idx"] = idx
    row["profile_missing"] = True
    return row


# ---------------------------------------------------------------------------------------------
# parsers


def tw20_str(v) -> str | None:
    """TwiBot-20 raw string -> stripped string; ``"None "`` / JSON null -> None; ``" "`` -> ''."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s == "None" else s


def tw20_bool(v) -> bool | None:
    s = tw20_str(v)
    if s is None or s == "":
        return None
    if s == "True":
        return True
    if s == "False":
        return False
    raise ValueError(f"not a TwiBot-20 bool: {v!r}")


def to_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s in ("", "None", "NULL", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def parse_twitter_date(v) -> datetime | None:
    """'Tue Nov 18 10:27:25 +0000 2008' (API v1.1, TwiBot-20/Cresci) or
    '2020-01-16 02:02:55+00:00' (TwiBot-22) -> aware UTC datetime; None/'None' -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "None", "NULL"):
        return None
    if s[:1].isdigit():
        dt = datetime.fromisoformat(s)
    else:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def cresci_bool(v) -> bool | None:
    """Cresci CSV flags: '1' -> True, '0' -> False, ''/NULL -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "NULL"):
        return None
    if s in ("1", "True", "true"):
        return True
    if s in ("0", "False", "false"):
        return False
    raise ValueError(f"not a Cresci bool: {v!r}")


def cresci_str(v) -> str | None:
    if v is None:
        return None
    s = str(v)
    return None if s == "NULL" else s
