#!/usr/bin/env python3
"""
Batch-parse TLE files (single or catalogs) using Skyfield and export metadata.

Features:
- Reads one or more TLE files. Each file may contain many satellites (name+2-line sets).
- Extracts core orbital elements and derived values (period, SMA, perigee/apogee).
- Outputs a CSV (default) or JSON Lines file.

Examples:
  # Parse a single multi-satellite catalog
  python tools/parse_tle_batch.py --in-file docs/DTC_tle.txt --out output/tle_catalog.csv

  # Parse multiple .tle files in a directory recursively
  python tools/parse_tle_batch.py --in-dir path/to/tle_dir --pattern "*.tle" --recursive \
      --out output/all_tles.jsonl --format jsonl

  # Use TLE from code/config.py as a quick test
  python tools/parse_tle_batch.py --from-config --out output/config_tle.csv

Dependencies:
  pip install skyfield sgp4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Dict


EARTH_RADIUS_KM = 6378.137  # WGS‑84 equatorial radius
MU_EARTH_KM3_S2 = 398600.4418


@dataclass
class TLEEntry:
    name: str
    l1: str
    l2: str
    source_file: Optional[str] = None


def _parse_epoch_from_l1(l1: str) -> Optional[datetime]:
    """Parse epoch from TLE line 1 (YYDDD.DDDDDDDD) as UTC datetime.

    Returns aware datetime in UTC, or None on failure.
    """
    try:
        # Columns per TLE spec: epoch year in columns 19-20, day of year (incl. fraction) in 21-32
        year_field = l1[18:20]
        day_field = l1[20:32]
        yy = int(year_field)
        # TLE epoch uses 1957-2056 2-digit years; usual convention: 57-99 -> 1900s, 00-56 -> 2000s
        year = 1900 + yy if yy >= 57 else 2000 + yy
        day_of_year = float(day_field)
        day_int = int(math.floor(day_of_year))
        day_frac = day_of_year - day_int
        base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int - 1)
        epoch = base + timedelta(days=day_frac)
        return epoch
    except Exception:
        return None


def _iter_tle_sets_from_lines(lines: Sequence[str], default_name: str) -> Iterator[TLEEntry]:
    """Yield TLEEntry from a list of lines. Accepts 2-line (no name) or 3-line sets.

    - If a name line precedes the 1/2 lines, it is used.
    - If no name line, uses default_name.
    - Ignores empty lines and comments starting with '#'.
    """
    buf: List[str] = []
    last_name: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        # A name line is anything that is not a TLE line starting with '1 ' or '2 '
        if not (line.startswith('1 ') or line.startswith('2 ')):
            last_name = line
            continue
        buf.append(line)
        if len(buf) == 2:
            if buf[0].startswith('1 ') and buf[1].startswith('2 '):
                name = last_name if last_name else default_name
                yield TLEEntry(name=name, l1=buf[0], l2=buf[1])
                buf.clear()
                last_name = None
            else:
                # Misaligned; reset and try to resync
                buf = [buf[-1]]
    # No partial handling needed; leftover is ignored


def load_tle_entries_from_file(path: str) -> List[TLEEntry]:
    with open(path, 'r') as f:
        lines = f.readlines()
    default_name = os.path.basename(path)
    entries = list(_iter_tle_sets_from_lines(lines, default_name))
    for e in entries:
        e.source_file = path
    return entries


def load_tle_entries_from_config() -> List[TLEEntry]:
    """Load a single TLE from code/config.py (tle_lines or tle_path)."""
    import sys
    sys.path.append('code')
    from config import CONFIG  # type: ignore
    tle_lines = CONFIG.get('tle_lines')
    tle_path = CONFIG.get('tle_path')
    tle_name = str(CONFIG.get('tle_name', 'SAT'))
    if (isinstance(tle_lines, (list, tuple)) and len(tle_lines) >= 2
            and isinstance(tle_lines[0], str) and isinstance(tle_lines[1], str)):
        return [TLEEntry(name=tle_name, l1=tle_lines[0].strip(), l2=tle_lines[1].strip(), source_file='config')]
    if tle_path and os.path.exists(tle_path):
        return load_tle_entries_from_file(tle_path)
    raise RuntimeError('No TLE provided in CONFIG (tle_lines or tle_path).')


def discover_tle_files(in_dir: str, pattern: str, recursive: bool) -> List[str]:
    base = os.path.abspath(in_dir)
    if recursive:
        files = glob.glob(os.path.join(base, '**', pattern), recursive=True)
    else:
        files = glob.glob(os.path.join(base, pattern))
    files = [p for p in files if os.path.isfile(p)]
    files.sort()
    return files


def parse_one_satellite(entry: TLEEntry) -> Dict[str, object]:
    try:
        from skyfield.api import EarthSatellite  # type: ignore
    except Exception as e:
        raise SystemExit("Skyfield/sgp4 not available. Please `pip install skyfield sgp4`.")

    sat = EarthSatellite(entry.l1, entry.l2, entry.name)

    # Elements from SGP4 model
    model = sat.model  # Satrec
    satnum = getattr(model, 'satnum', None)
    intldesg = getattr(model, 'intldesg', None)
    ecco = float(getattr(model, 'ecco', float('nan')))
    inclo_rad = float(getattr(model, 'inclo', float('nan')))
    nodeo_rad = float(getattr(model, 'nodeo', float('nan')))
    argpo_rad = float(getattr(model, 'argpo', float('nan')))
    mo_rad = float(getattr(model, 'mo', float('nan')))
    no_kozai = float(getattr(model, 'no_kozai', float('nan')))  # rad/min
    bstar = float(getattr(model, 'bstar', float('nan')))

    mean_motion_rev_per_day = (no_kozai * 1440.0) / (2.0 * math.pi) if math.isfinite(no_kozai) else float('nan')
    period_min = 1440.0 / mean_motion_rev_per_day if mean_motion_rev_per_day and mean_motion_rev_per_day > 0 else float('nan')

    # Derive SMA and apo/perigee altitudes (km)
    if math.isfinite(mean_motion_rev_per_day) and mean_motion_rev_per_day > 0:
        n_rad_s = (mean_motion_rev_per_day * 2.0 * math.pi) / 86400.0
        a_km = (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)
    else:
        a_km = float('nan')
    perigee_km = a_km * (1.0 - ecco) - EARTH_RADIUS_KM if math.isfinite(a_km) and math.isfinite(ecco) else float('nan')
    apogee_km = a_km * (1.0 + ecco) - EARTH_RADIUS_KM if math.isfinite(a_km) and math.isfinite(ecco) else float('nan')

    # Epoch (prefer Skyfield conversion; fall back to manual parse)
    try:
        epoch_dt = sat.epoch.utc_datetime().replace(tzinfo=timezone.utc)
    except Exception:
        epoch_dt = _parse_epoch_from_l1(entry.l1)
    epoch_iso = epoch_dt.isoformat().replace('+00:00', 'Z') if isinstance(epoch_dt, datetime) else ''

    row: Dict[str, object] = {
        'name': entry.name,
        'satnum': satnum,
        'intl_des': intldesg,
        'epoch_utc': epoch_iso,
        'incl_deg': math.degrees(inclo_rad) if math.isfinite(inclo_rad) else float('nan'),
        'raan_deg': math.degrees(nodeo_rad) if math.isfinite(nodeo_rad) else float('nan'),
        'ecc': ecco,
        'argp_deg': math.degrees(argpo_rad) if math.isfinite(argpo_rad) else float('nan'),
        'mean_anom_deg': math.degrees(mo_rad) if math.isfinite(mo_rad) else float('nan'),
        'mean_motion_rev_per_day': mean_motion_rev_per_day,
        'period_min': period_min,
        'sma_km': a_km,
        'perigee_km': perigee_km,
        'apogee_km': apogee_km,
        'bstar': bstar,
        'source_file': entry.source_file or '',
        'tle_line1': entry.l1,
        'tle_line2': entry.l2,
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Batch parse TLE files/catalogs using Skyfield.')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--in-file', type=str, help='Path to a TLE file (may contain many satellites).')
    src.add_argument('--in-dir', type=str, help='Directory to search for TLE files.')
    src.add_argument('--from-config', action='store_true', help='Use TLE from code/config.py (tle_lines or tle_path).')
    p.add_argument('--pattern', type=str, default='*.tle', help='Glob pattern when using --in-dir (default: *.tle).')
    p.add_argument('--recursive', action='store_true', help='Recurse into subdirectories when using --in-dir.')
    p.add_argument('--format', type=str, default='csv', choices=['csv', 'jsonl'], help='Output format (csv or jsonl).')
    p.add_argument('--out', type=str, default='output/tle_batch.csv', help='Output file path.')
    p.add_argument('--dedup', action='store_true', help='Deduplicate by satnum+epoch if multiple entries found.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Collect entries
    entries: List[TLEEntry] = []
    if args.from_config:
        entries = load_tle_entries_from_config()
    elif args.in_file:
        entries = load_tle_entries_from_file(args.in_file)
    else:
        files = discover_tle_files(args.in_dir, args.pattern, args.recursive)
        if not files:
            raise SystemExit('No files found matching pattern.')
        for fp in files:
            entries.extend(load_tle_entries_from_file(fp))

    if not entries:
        print('No TLE entries discovered.')
        return

    print(f'Discovered {len(entries)} TLE entries. Parsing with Skyfield...')
    rows: List[Dict[str, object]] = []
    for e in entries:
        try:
            rows.append(parse_one_satellite(e))
        except SystemExit:
            raise
        except Exception as ex:
            print(f'Warning: failed to parse {e.name} from {e.source_file}: {ex}')

    if args.dedup:
        seen = set()
        unique_rows = []
        for r in rows:
            key = (r.get('satnum'), r.get('epoch_utc'))
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(r)
        rows = unique_rows
        print(f'Deduplicated to {len(rows)} entries by satnum+epoch.')

    # Write output
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.format == 'csv':
        # Define a stable header order
        header = [
            'name', 'satnum', 'intl_des', 'epoch_utc',
            'incl_deg', 'raan_deg', 'ecc', 'argp_deg', 'mean_anom_deg',
            'mean_motion_rev_per_day', 'period_min', 'sma_km', 'perigee_km', 'apogee_km',
            'bstar', 'source_file', 'tle_line1', 'tle_line2',
        ]
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in header})
        print(f'Saved CSV: {out_path} ({len(rows)} rows)')
    else:
        with open(out_path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'Saved JSONL: {out_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()

