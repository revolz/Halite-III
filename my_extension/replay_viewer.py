#!/usr/bin/env python3
"""
replay_viewer.py – Python tkinter viewer for Halite III .hlt replay files.

Uses the same graphics resources (sprites, color palette) as the original
libhaliteviz visualizer (TheSea / Turtles theme).

Usage:
    python replay_viewer.py <replay.hlt>
    python replay_viewer.py replays/replay-*.hlt   # first matching file

Controls:
    ←/→ arrow keys   Step one turn backward / forward
    Space            Play / Pause
    Home/End         Jump to first / last turn
    Mouse wheel      Scroll through turns
"""

import json
import math
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageTk


# ---------------------------------------------------------------------------
# Paths to libhaliteviz assets (relative to this file's directory)
# ---------------------------------------------------------------------------

_MY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MY_DIR)
_ASSETS_DIR = os.path.join(_REPO_ROOT, 'libhaliteviz', 'assets')
_ASSETS_RAW = os.path.join(_ASSETS_DIR, 'raw')
_COLORS_JS = os.path.join(_REPO_ROOT, 'libhaliteviz', 'src', 'colors.js')


# ---------------------------------------------------------------------------
# TheSea / Turtles theme — exact values from libhaliteviz/src/colors.js
#   and libhaliteviz/src/theme.js
# ---------------------------------------------------------------------------

# Background colour for the "TheSea" colour theme (used by default "Turtles" theme)
BG_HEX = 0x141EA0
BG_RGB = ((BG_HEX >> 16) & 0xFF, (BG_HEX >> 8) & 0xFF, BG_HEX & 0xFF)  # (20, 30, 160)
BG_COLOR = f'#{BG_HEX:06x}'

# Player colours for the "Turtles" theme.
# Player 1 uses the purple scheme (swapped with red, which moves to slot 3) so
# the second bot renders purple instead of red.
_PLAYER_COLORS_HEX = [0x1BB15A, 0xffa2e2, 0xFABB2C, 0xF54356]
PLAYER_COLORS_HEX = _PLAYER_COLORS_HEX
PLAYER_COLORS = [f'#{c:06x}' for c in _PLAYER_COLORS_HEX]

PANEL_BG = '#0d1020'
TEXT_COLOR = '#dddddd'

# Playback constants
TICK_MS = 33               # ~30 fps update interval
TURNS_PER_SEC_DEFAULT = 3.0  # default playback speed

# Direction → PIL rotation angle (counter-clockwise degrees).
# Turtle sprites face NORTH by default (matching PIXI rotation=0 → N).
_DIR_TO_ANGLE: Dict[str, int] = {'n': 0, 'e': -90, 's': 180, 'w': 90, 'o': 0}

# Human-readable direction labels
_DIR_LABELS: Dict[str, str] = {
    'n': '▲ North', 's': '▼ South', 'e': '▶ East', 'w': '◀ West', 'o': '● Stay (mine)',
}

# Attack/shockwave ring colour: light grey for same-owner, white for multi-owner
_EXPLOSION_RGB_DEFAULT = (0xFF, 0xFF, 0xFF)

# Per-player sprite prefixes (Turtles theme).  Player 1 ↔ purple (red moved to 3).
_PLAYER_SPRITE_NAMES = ['green', 'purple', 'yellow', 'red']
# Corresponding base sprite filenames in assets/ (same red↔purple swap)
_BASE_SPRITE_FILES = [
    os.path.join(_ASSETS_DIR, 'base-green.png'),
    os.path.join(_ASSETS_DIR, 'base-purple.png'),
    os.path.join(_ASSETS_DIR, 'base-yellow.png'),
    os.path.join(_ASSETS_DIR, 'base-red.png'),
]
# Turtle ship sprites: 3 load levels per player
_TURTLE_SPRITE_FILES = [
    [os.path.join(_ASSETS_RAW, f'turtle-{name}-{lvl}.png')
     for lvl in ('low', 'medium', 'high')]
    for name in _PLAYER_SPRITE_NAMES
]
_HALO_FILE = os.path.join(_ASSETS_DIR, 'halo2.png')


# ---------------------------------------------------------------------------
# Load COLOR_SCALE from the original colors.js
# ---------------------------------------------------------------------------

def _load_color_scale() -> List[int]:
    """Parse the COLOR_SCALE array from libhaliteviz/src/colors.js."""
    try:
        with open(_COLORS_JS, 'r') as f:
            src = f.read()
        # Match the first array assigned to COLOR_SCALE (stops at ]; )
        m = re.search(r'export const COLOR_SCALE\s*=\s*\[(.*?)\]', src, re.DOTALL)
        if m:
            return [int(v, 16) for v in re.findall(r'0x([0-9a-fA-F]+)', m.group(1))]
    except Exception:
        pass
    # Fallback: linear interpolation from blue to cyan
    start = (0x2c, 0x82, 0xfd)
    end = (0xb3, 0xfe, 0xf5)
    return [
        (round(start[0] + i / 999 * (end[0] - start[0])) << 16) |
        (round(start[1] + i / 999 * (end[1] - start[1])) << 8) |
        round(start[2] + i / 999 * (end[2] - start[2]))
        for i in range(1000)
    ]

COLOR_SCALE = _load_color_scale()


def _hex_to_rgb(h: int) -> Tuple[int, int, int]:
    return (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF


def _alpha_blend(rgb1: Tuple, rgb2: Tuple, alpha: float) -> Tuple[int, int, int]:
    """PIXI-style alpha blend: alpha * rgb1 + (1-alpha) * rgb2."""
    return (
        int(alpha * rgb1[0] + (1 - alpha) * rgb2[0]),
        int(alpha * rgb1[1] + (1 - alpha) * rgb2[1]),
        int(alpha * rgb1[2] + (1 - alpha) * rgb2[2]),
    )


def halite_to_rgb(halite: int, max_production: int = 1000) -> Tuple[int, int, int]:
    """
    Map halite → RGB using the original COLOR_SCALE, then alpha-blend with
    the TheSea background (PIXI cell.alpha = 0.9).
    Matches map.js productionToColor() + alphaBlend().
    """
    f = min(1.0, halite / max_production)
    idx = int(f * (len(COLOR_SCALE) - 1))
    base = _hex_to_rgb(COLOR_SCALE[idx])
    return _alpha_blend(base, BG_RGB, 0.9)


def halite_cell_fraction(halite: int, max_production: int = 1000) -> float:
    """
    Return the fraction of the cell to fill, matching map.js update() sizing:
      f = min(1.25, halite / max_production)
      if f <= 0.2: size = sqrt(f)
      else:        size = 0.3 + 0.7 * f
    """
    f = min(1.25, halite / max_production)
    if f <= 0.0:
        return 0.0
    if f <= 0.2:
        return math.sqrt(f)
    return 0.3 + 0.7 * f


# ---------------------------------------------------------------------------
# Sprite cache
# ---------------------------------------------------------------------------

class SpriteCache:
    """Load and cache PIL images for ships, bases, and the halo."""

    def __init__(self):
        self._raw_ships: List[List[Optional[Image.Image]]] = []
        self._raw_bases: List[Optional[Image.Image]] = []
        self._raw_halo: Optional[Image.Image] = None
        self._ship_cache: dict = {}
        self._rotated_cache: dict = {}
        self._base_cache: dict = {}
        self._halo_cache: dict = {}
        self._load_raw()

    def _load(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert('RGBA')
        except Exception:
            return None

    def _load_raw(self):
        self._raw_ships = [
            [self._load(p) for p in player_files]
            for player_files in _TURTLE_SPRITE_FILES
        ]
        self._raw_bases = [self._load(p) for p in _BASE_SPRITE_FILES]
        self._raw_halo = self._load(_HALO_FILE)

    def _resize(self, img: Optional[Image.Image], size: int) -> Optional[Image.Image]:
        if img is None or size <= 0:
            return None
        return img.resize((size, size), Image.LANCZOS)

    def ship(self, player: int, energy: int, max_energy: int, cell_size: int) -> Optional[Image.Image]:
        f = energy / max_energy if max_energy > 0 else 0
        level = 0 if f < 0.25 else (1 if f < 0.75 else 2)
        size = max(1, int(1.5 * cell_size))
        key = (player, level, size)
        if key not in self._ship_cache:
            sprites = self._raw_ships[player % len(self._raw_ships)]
            self._ship_cache[key] = self._resize(sprites[level], size)
        return self._ship_cache[key]

    def ship_rotated(self, player: int, energy: int, max_energy: int,
                     cell_size: int, angle: int) -> Optional[Image.Image]:
        """Return the ship sprite rotated by *angle* degrees (PIL CCW convention)."""
        f = energy / max_energy if max_energy > 0 else 0
        level = 0 if f < 0.25 else (1 if f < 0.75 else 2)
        size = max(1, int(1.5 * cell_size))
        key = (player, level, size, angle)
        if key not in self._rotated_cache:
            base = self.ship(player, energy, max_energy, cell_size)
            if base is None:
                self._rotated_cache[key] = None
            elif angle == 0:
                self._rotated_cache[key] = base
            else:
                self._rotated_cache[key] = base.rotate(angle, expand=True,
                                                        resample=Image.BILINEAR)
        return self._rotated_cache[key]

    def base(self, player: int, cell_size: int, is_dropoff: bool = False) -> Optional[Image.Image]:
        scale = 1.5 if is_dropoff else 2.0
        size = max(1, int(scale * cell_size))
        key = (player, size)
        if key not in self._base_cache:
            raw = self._raw_bases[player % len(self._raw_bases)]
            self._base_cache[key] = self._resize(raw, size)
        return self._base_cache[key]

    def halo(self, cell_size: int) -> Optional[Image.Image]:
        size = max(1, int(1.5 * cell_size))
        if size not in self._halo_cache:
            self._halo_cache[size] = self._resize(self._raw_halo, size)
        return self._halo_cache[size]


# ---------------------------------------------------------------------------
# Replay parser
# ---------------------------------------------------------------------------

def load_replay(path: str) -> dict:
    """Load a .hlt file (zstd-compressed or plain JSON)."""
    raw = open(path, 'rb').read()
    try:
        import zstd
        data = zstd.decompress(raw)
    except (ImportError, Exception):
        data = raw
    return json.loads(data)


def build_display_states(replay: dict) -> List[dict]:
    """
    Convert replay frames into a list of display states.

    State 0 = initial (before any turns).
    State N = after turn N completes.

    For each state we store:
      - halite_map[y][x]
      - ships: {pid: {sid: {'x','y','energy','is_inspired'}}}
      - structures: {pid: [(x,y,'factory'|'dropoff'), ...]}
      - energy: {pid: int}
      - turn: int (0 = initial)
      - events: list of event dicts for this turn
    """
    width = replay['production_map']['width']
    height = replay['production_map']['height']
    full_frames = replay['full_frames']
    players = replay['players']
    num_turns = replay['game_statistics']['number_turns']

    # Build initial halite map
    grid = replay['production_map']['grid']
    halite_map = [[grid[y][x]['energy'] for x in range(width)]
                  for y in range(height)]

    # Initial player structures (factories only)
    structures: Dict[int, List[Tuple[int, int, str]]] = {}
    for p in players:
        pid = p['player_id']
        fx, fy = p['factory_location']['x'], p['factory_location']['y']
        structures[pid] = [(fx, fy, 'factory')]

    # Initial player energies
    energy = {p['player_id']: p['energy'] for p in players}

    # Initial display state (before any turns)
    display_states = [{
        'halite_map': [row[:] for row in halite_map],
        'ships': {p['player_id']: {} for p in players},
        'structures': {pid: list(v) for pid, v in structures.items()},
        'energy': dict(energy),
        'turn': 0,
        'events': [],
        'ship_owner_map': {},
    }]

    # Build running halite map by applying cell deltas per frame
    # Also extract end-of-turn energy and track structures
    for turn_idx in range(1, num_turns + 1):
        if turn_idx >= len(full_frames):
            break
        frame = full_frames[turn_idx]

        # Apply cell halite changes for this turn
        for cell in frame.get('cells', []):
            cx, cy = cell['x'], cell['y']
            halite_map[cy][cx] = cell['production']

        # Track new dropoffs from construct events
        for evt in frame.get('events', []):
            if evt.get('type') == 'construct':
                pid = evt.get('owner_id')
                lx = evt['location']['x']
                ly = evt['location']['y']
                if pid is not None and pid in structures:
                    structures[pid].append((lx, ly, 'dropoff'))

        # End-of-turn energy
        frame_energy = frame.get('energy', {})
        for pid_str, e in frame_energy.items():
            energy[int(pid_str)] = e

        # Ships to display after turn_idx: use start-of-(turn_idx+1) entities
        # (= where ships are after turn turn_idx completed)
        next_turn = turn_idx + 1
        if next_turn < len(full_frames):
            raw_ents = full_frames[next_turn].get('entities', {})
        else:
            # Final turn: no next frame exists. Start from this frame's entities
            # and remove any ships destroyed in shipwreck events this turn.
            destroyed_ids: set = set()
            for evt in frame.get('events', []):
                if evt.get('type') == 'shipwreck':
                    for s in evt.get('ships', []):
                        if isinstance(s, int):
                            destroyed_ids.add(s)
                        elif isinstance(s, dict) and 'id' in s:
                            destroyed_ids.add(int(s['id']))
            if destroyed_ids:
                raw_ents = {}
                for pid_str, sdict in frame.get('entities', {}).items():
                    raw_ents[pid_str] = {
                        sid_str: sdata for sid_str, sdata in sdict.items()
                        if int(sid_str) not in destroyed_ids
                    }
            else:
                raw_ents = frame.get('entities', {})

        ships: Dict[int, dict] = {}
        for pid_str, ship_dict in raw_ents.items():
            pid = int(pid_str)
            ships[pid] = {}
            for sid_str, sdata in ship_dict.items():
                ships[pid][int(sid_str)] = {
                    'x': sdata['x'], 'y': sdata['y'],
                    'energy': sdata.get('energy', 0),
                    'is_inspired': sdata.get('is_inspired', False),
                }

        # Build ship_id→owner map from THIS frame's entities (before the turn runs).
        # These are exactly the ships that may appear in collision events for this turn.
        prev_ents = frame.get('entities', {})
        ship_owner_map: Dict[int, int] = {}
        for pid_str, ship_dict in prev_ents.items():
            for sid_str in ship_dict:
                ship_owner_map[int(sid_str)] = int(pid_str)

        display_states.append({
            'halite_map': [row[:] for row in halite_map],
            'ships': ships,
            'structures': {pid: list(v) for pid, v in structures.items()},
            'energy': dict(energy),
            'turn': turn_idx,
            'events': list(frame.get('events', [])),
            'moves': {},  # filled in post-processing pass below
            'ship_owner_map': ship_owner_map,  # ship_id→owner for event colouring
        })

    # Post-processing: attach moves that transition state[i] → state[i+1]
    # Those moves live in full_frames[i+1] (they execute AT turn i+1,
    # sliding ships from state[i] positions to state[i+1] positions).
    for i in range(len(display_states)):
        frame_idx = i + 1
        if frame_idx < len(full_frames):
            display_states[i]['moves'] = _parse_moves(
                full_frames[frame_idx].get('moves', {}))

    return display_states


def _parse_moves(raw_moves: dict) -> Dict[int, Dict[int, str]]:
    """Convert replay moves dict to {player_id: {ship_id: direction}} mapping."""
    moves: Dict[int, Dict[int, str]] = {}
    for pid_str, cmds in raw_moves.items():
        pid = int(pid_str)
        moves[pid] = {}
        if isinstance(cmds, list):
            for cmd in cmds:
                if cmd.get('type') == 'm':
                    moves[pid][int(cmd['id'])] = cmd.get('direction', 'o')
                elif cmd.get('type') == 'c' and 'id' in cmd:
                    moves[pid][int(cmd['id'])] = 'o'
        elif isinstance(cmds, dict):
            for _k, cmd in cmds.items():
                if cmd.get('type') == 'm':
                    moves[pid][int(cmd['id'])] = cmd.get('direction', 'o')
                elif cmd.get('type') == 'c' and 'id' in cmd:
                    moves[pid][int(cmd['id'])] = 'o'
    return moves


def _cubic_ease(t: float) -> float:
    """Cubic ease-in-out matching sprite.js interpolation logic."""
    s = t / 0.5
    if s < 1:
        return s * s * s / 2
    s -= 2
    return (s * s * s + 2) / 2


def _draw_event_rings(img: Image.Image, events: list, cell_size: int,
                      time: float, ship_owner_map: Optional[Dict[int, int]] = None) -> None:
    """
    Draw shockwave rings for shipwreck/spawn events onto *img* (in-place).
    Ring expands and fades as time goes 0→1 (mimics PIXI ShockwaveFilter).

    *ship_owner_map*: {ship_id: player_id} built from state['ships'] so we can
    colour single-owner collisions with the player colour.
    The underlying replay format stores ships as bare integer IDs (matching the
    original C++ CollisionEvent::to_json which serialises vector<Entity::id_type>).
    """
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    drew_any = False

    for evt in events:
        etype = evt.get('type', '')
        if etype not in ('shipwreck', 'spawn'):
            continue
        loc = evt.get('location')
        if not loc:
            continue
        ex, ey = loc['x'], loc['y']
        cx = ex * cell_size + cell_size // 2
        cy = ey * cell_size + cell_size // 2

        # Expanding ring
        ring_r = max(2, int((0.3 + time * 2.2) * cell_size))
        ring_alpha = int((1.0 - time) * 230)
        if ring_alpha <= 0:
            continue

        # Colour: white for multi-player collision; player colour for same-owner
        if etype == 'spawn':
            color_rgb = (255, 255, 255)
        else:
            # ships field = list of bare integer ship IDs (original C++ format)
            ship_ids = evt.get('ships', [])
            owners: set = set()
            if ship_owner_map:
                for sid in ship_ids:
                    if isinstance(sid, int) and sid in ship_owner_map:
                        owners.add(ship_owner_map[sid])
            if len(owners) == 1:
                c = _PLAYER_COLORS_HEX[next(iter(owners)) % len(_PLAYER_COLORS_HEX)]
                color_rgb = _hex_to_rgb(c)
            else:
                color_rgb = (255, 255, 255)

        ring_width = max(2, cell_size // 3)
        box = (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r)
        draw.ellipse(box, outline=(*color_rgb, ring_alpha), width=ring_width)

        # Inner glow that fades quickly
        if time < 0.4:
            glow_r = max(1, int(cell_size * 0.45))
            glow_alpha = int((1.0 - time / 0.4) * 180)
            gb = (cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r)
            draw.ellipse(gb, fill=(*color_rgb, glow_alpha))

        drew_any = True

    if drew_any:
        composited = Image.alpha_composite(img.convert('RGBA'), overlay)
        img.paste(composited.convert('RGB'), (0, 0))


# ---------------------------------------------------------------------------
# Frame renderer — produces a PIL Image matching the original TheSea theme
# ---------------------------------------------------------------------------

def render_frame(state: dict, map_w: int, map_h: int,
                 cell_size: int, max_production: int,
                 sprites: SpriteCache, max_energy: int = 1000,
                 time: float = 0.0,
                 next_state: Optional[dict] = None) -> Image.Image:
    """
    Render a single display state to a PIL Image.
    *time* (0.0–1.0): sub-frame position for smooth interpolation.
    *next_state*: required when time > 0 for ship position interpolation.
    """
    pw = map_w * cell_size
    ph = map_h * cell_size
    img = Image.new('RGB', (pw, ph), BG_RGB)

    # --- Halite cells ---
    hmap = state['halite_map']
    for y in range(map_h):
        for x in range(map_w):
            h = hmap[y][x]
            if h <= 0:
                continue
            frac = halite_cell_fraction(h, max_production)
            if frac <= 0:
                continue
            side = max(1, int(frac * cell_size))
            rgb = halite_to_rgb(h, max_production)
            cx = x * cell_size + cell_size // 2
            cy = y * cell_size + cell_size // 2
            x0 = cx - side // 2
            y0 = cy - side // 2
            cell_img = Image.new('RGB', (side, side), rgb)
            img.paste(cell_img, (x0, y0))

    # --- Factories and dropoffs (drawn under ships) ---
    for pid, structs in state['structures'].items():
        for sx, sy, stype in structs:
            is_dropoff = (stype == 'dropoff')
            spr = sprites.base(pid, cell_size, is_dropoff)
            if spr is None:
                continue
            cx = sx * cell_size + cell_size // 2
            cy = sy * cell_size + cell_size // 2
            ox = cx - spr.width // 2
            oy = cy - spr.height // 2
            img.paste(spr, (ox, oy), mask=spr)

    # --- Event animations (shockwave rings for shipwreck / spawn) ---
    events = state.get('events', [])
    if events:
        # ship_owner_map built at parse-time from entities at START of this turn,
        # so it contains the ships that may have collided (bare int IDs per C++ format).
        _draw_event_rings(img, events, cell_size, time,
                          state.get('ship_owner_map', {}))

    # --- Cubic ease-in-out for sub-frame ship interpolation ---
    interp_t = 0.0
    if time > 0.0 and next_state is not None:
        interp_t = _cubic_ease(min(1.0, time))

    # --- Ships ---
    moves = state.get('moves', {})
    for pid, ship_dict in state['ships'].items():
        pid_moves = moves.get(pid, {})
        next_ships = next_state['ships'].get(pid, {}) if next_state else {}

        for sid, ship in ship_dict.items():
            # Compute draw position (interpolate toward next_state)
            draw_x = float(ship['x'])
            draw_y = float(ship['y'])
            if interp_t > 0.0 and sid in next_ships:
                ns = next_ships[sid]
                dx = ns['x'] - ship['x']
                dy = ns['y'] - ship['y']
                # Wraparound correction
                if dx > map_w // 2:
                    dx -= map_w
                elif dx < -(map_w // 2):
                    dx += map_w
                if dy > map_h // 2:
                    dy -= map_h
                elif dy < -(map_h // 2):
                    dy += map_h
                draw_x = ship['x'] + dx * interp_t
                draw_y = ship['y'] + dy * interp_t

            cx = int(draw_x * cell_size + cell_size // 2)
            cy = int(draw_y * cell_size + cell_size // 2)

            # Halo for inspired ships
            if ship.get('is_inspired'):
                halo = sprites.halo(cell_size)
                if halo is not None:
                    ox = cx - halo.width // 2
                    oy = cy - halo.height // 2
                    img.paste(halo, (ox, oy), mask=halo)

            # Ship sprite with rotation toward movement direction
            direction = pid_moves.get(sid, 'o')
            angle = _DIR_TO_ANGLE.get(direction, 0)
            spr = sprites.ship_rotated(pid, ship['energy'], max_energy, cell_size, angle)
            if spr is None:
                continue
            ox = cx - spr.width // 2
            oy = cy - spr.height // 2
            img.paste(spr, (ox, oy), mask=spr)

    return img


# ---------------------------------------------------------------------------
# Viewer application
# ---------------------------------------------------------------------------

class HaliteViewer:

    def __init__(self, root: tk.Tk, replay_path: str):
        self.root = root
        self.root.title(f"Halite III – {os.path.basename(replay_path)}")
        self.root.configure(bg=PANEL_BG)

        replay = load_replay(replay_path)
        self.map_w = replay['production_map']['width']
        self.map_h = replay['production_map']['height']
        self.players = replay['players']
        self.display_states = build_display_states(replay)
        self.num_states = len(self.display_states)
        self.player_names = {p['player_id']: p['name'] for p in self.players}
        gc = replay.get('GAME_CONSTANTS', {})
        self.max_production = gc.get('MAX_CELL_PRODUCTION', 1000)
        self.max_energy = gc.get('MAX_ENERGY', 1000)

        # Cell size: scale map to ~600px canvas, matching original 600×600 target
        target = 600
        self.cell_size = max(6, min(24, target // max(self.map_w, self.map_h)))

        self.sprites = SpriteCache()

        self.current = 0
        self.playing = False
        self._after_id: Optional[str] = None
        self.subframe_time: float = 0.0
        self.turns_per_second: float = TURNS_PER_SEC_DEFAULT
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._draw_state(0)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        canvas_w = self.map_w * self.cell_size
        canvas_h = self.map_h * self.cell_size

        self.canvas = tk.Canvas(
            self.root, width=canvas_w, height=canvas_h,
            bg=BG_COLOR, highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP, padx=4, pady=4)
        # Single image item — updated every frame
        self._img_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Controls panel
        panel = tk.Frame(self.root, bg=PANEL_BG)
        panel.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        self.turn_label = tk.Label(
            panel, text='Turn 0', fg=TEXT_COLOR, bg=PANEL_BG,
            font=('Consolas', 11, 'bold'), width=12,
        )
        self.turn_label.grid(row=0, column=0, padx=6)

        btn_style = dict(bg='#1a1e3a', fg=TEXT_COLOR, relief=tk.FLAT,
                         font=('Consolas', 12), padx=8)
        tk.Button(panel, text='◀◀', command=self._go_first, **btn_style).grid(row=0, column=1)
        tk.Button(panel, text='◀',  command=self._prev,     **btn_style).grid(row=0, column=2)
        self.play_btn = tk.Button(panel, text='▶', command=self._toggle_play, **btn_style)
        self.play_btn.grid(row=0, column=3)
        tk.Button(panel, text='▶',  command=self._next,     **btn_style).grid(row=0, column=4)
        tk.Button(panel, text='▶▶', command=self._go_last,  **btn_style).grid(row=0, column=5)

        self.slider = ttk.Scale(
            panel, from_=0, to=self.num_states - 1,
            orient=tk.HORIZONTAL, command=self._on_slider, length=280,
        )
        self.slider.grid(row=0, column=6, padx=8)

        tk.Label(panel, text='Speed:', fg=TEXT_COLOR, bg=PANEL_BG,
                 font=('Consolas', 10)).grid(row=0, column=7)
        self.speed_slider = ttk.Scale(
            panel, from_=20, to=600, orient=tk.HORIZONTAL,
            command=self._on_speed, length=100,
        )
        self.speed_slider.set(120)
        self.speed_slider.grid(row=0, column=8, padx=4)

        # Score panel
        score_frame = tk.Frame(self.root, bg=PANEL_BG)
        score_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4)
        self.score_labels = {}
        for i, p in enumerate(self.players):
            pid = p['player_id']
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
            lbl = tk.Label(score_frame, text=f'P{pid}', fg=color, bg=PANEL_BG,
                           font=('Consolas', 11, 'bold'))
            lbl.grid(row=0, column=i * 2, padx=(10, 2))
            val = tk.Label(score_frame, text='0', fg=TEXT_COLOR, bg=PANEL_BG,
                           font=('Consolas', 11))
            val.grid(row=0, column=i * 2 + 1, padx=(0, 10))
            self.score_labels[pid] = val

        self.root.bind('<Left>',  lambda e: self._prev())
        self.root.bind('<Right>', lambda e: self._next())
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<Home>',  lambda e: self._go_first())
        self.root.bind('<End>',   lambda e: self._go_last())
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Motion>',     self._on_canvas_motion)
        self.canvas.bind('<Leave>',      lambda e: self._hide_tooltip())

        # Floating tooltip window (hidden until a ship is hovered)
        self._tooltip_win = tk.Toplevel(self.root)
        self._tooltip_win.wm_overrideredirect(True)   # no title bar / border
        self._tooltip_win.withdraw()
        self._tooltip_lbl = tk.Label(
            self._tooltip_win,
            text='', justify=tk.LEFT, anchor=tk.NW,
            bg='#1a1e3a', fg=TEXT_COLOR,
            font=('Consolas', 10),
            padx=8, pady=6,
            relief=tk.SOLID, bd=1,
        )
        self._tooltip_lbl.pack()
        self._last_tooltip_cell: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ Drawing

    def _draw_state(self, idx: int, subframe: float = 0.0):
        idx = max(0, min(idx, self.num_states - 1))
        self.current = idx
        state = self.display_states[idx]

        next_state = None
        if subframe > 0.0 and idx + 1 < self.num_states:
            next_state = self.display_states[idx + 1]

        pil_img = render_frame(
            state, self.map_w, self.map_h,
            self.cell_size, self.max_production,
            self.sprites, self.max_energy,
            time=subframe, next_state=next_state,
        )
        # Keep a reference so tkinter doesn't garbage-collect it
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self._img_item, image=self._tk_image)

        # Update scores (halite bank + ship count)
        for pid, lbl in self.score_labels.items():
            e = state['energy'].get(pid, 0)
            name = self.player_names.get(pid, f'P{pid}')
            n_ships = len(state['ships'].get(pid, {}))
            lbl.config(text=f'{name}: {e:,} halite | {n_ships} ship{"s" if n_ships != 1 else ""}')

        total = self.num_states - 1
        self.turn_label.config(text=f'Turn {state["turn"]}/{total}')
        self.slider.set(idx)

    # ------------------------------------------------------------------ Tooltip

    def _on_canvas_motion(self, event):
        """Show a floating tooltip with ship info when hovering over a ship."""
        cx = event.x // self.cell_size
        cy = event.y // self.cell_size
        cell = (cx, cy)

        # Avoid re-computing if still on the same cell
        if cell == self._last_tooltip_cell:
            return
        self._last_tooltip_cell = cell

        state = self.display_states[self.current]
        moves = state.get('moves', {})

        # Search all ships for one at this map cell
        found = None
        for pid, ship_dict in state['ships'].items():
            for sid, ship in ship_dict.items():
                if ship['x'] == cx and ship['y'] == cy:
                    found = (pid, sid, ship)
                    break
            if found:
                break

        if found is None:
            self._hide_tooltip()
            return

        pid, sid, ship = found
        name = self.player_names.get(pid, f'P{pid}')
        direction = moves.get(pid, {}).get(sid, 'o')
        dir_label = _DIR_LABELS.get(direction, direction)
        cargo = ship['energy']
        cargo_pct = int(100 * cargo / max(1, self.max_energy))
        inspired = '⚡ Yes' if ship.get('is_inspired') else 'No'
        color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]

        text = (
            f'Ship #{sid}\n'
            f'Owner:   {name}  (P{pid})\n'
            f'Cargo:   {cargo:,} / {self.max_energy:,}  ({cargo_pct}%)\n'
            f'Action:  {dir_label}\n'
            f'Inspired:{inspired}'
        )
        self._tooltip_lbl.config(text=text, fg=color)

        # Position tooltip near cursor (but keep it on-screen)
        win_x = self.root.winfo_rootx() + event.x + 16
        win_y = self.root.winfo_rooty() + event.y + 16
        self._tooltip_win.wm_geometry(f'+{win_x}+{win_y}')
        self._tooltip_win.deiconify()
        self._tooltip_win.lift()

    def _hide_tooltip(self):
        self._tooltip_win.withdraw()
        self._last_tooltip_cell = None

    # ------------------------------------------------------------------ Navigation

    def _prev(self):
        self._pause()
        self._draw_state(self.current - 1)

    def _next(self):
        self._pause()
        self._draw_state(self.current + 1)

    def _go_first(self):
        self._pause()
        self._draw_state(0)

    def _go_last(self):
        self._pause()
        self._draw_state(self.num_states - 1)

    def _on_slider(self, value):
        idx = int(float(value))
        if idx != self.current:
            self._pause()
            self._draw_state(idx)

    def _on_speed(self, value):
        # Map slider 20→600 to turns_per_second 0.3→12
        v = float(value)
        self.turns_per_second = max(0.3, v / 50.0)

    def _on_wheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self._pause()
        self._draw_state(self.current + delta)

    # ------------------------------------------------------------------ Playback

    def _toggle_play(self):
        if self.playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.current >= self.num_states - 1:
            self.current = 0
            self.subframe_time = 0.0
            self._draw_state(0)
        self.playing = True
        self.play_btn.config(text='⏸')
        self._schedule_next()

    def _pause(self):
        self.playing = False
        self.subframe_time = 0.0
        self.play_btn.config(text='▶')
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_next(self):
        if self.playing:
            self._after_id = self.root.after(TICK_MS, self._play_step)

    def _play_step(self):
        self._after_id = None
        if not self.playing:
            return
        if self.current >= self.num_states - 1:
            self._pause()
            return
        # Advance subframe by turns_per_second × tick duration
        self.subframe_time += TICK_MS / 1000.0 * self.turns_per_second
        while self.subframe_time >= 1.0:
            self.subframe_time -= 1.0
            self.current += 1
            if self.current >= self.num_states - 1:
                self.current = self.num_states - 1
                self.subframe_time = 0.0
                self._draw_state(self.current)
                self._pause()
                return
        self._draw_state(self.current, subframe=self.subframe_time)
        self._schedule_next()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.resizable(False, False)

    if len(sys.argv) > 1:
        replay_path = sys.argv[1]
        # Support glob-style: if path contains * expand it
        if '*' in replay_path:
            import glob
            matches = sorted(glob.glob(replay_path))
            if not matches:
                print(f'No files matching: {replay_path}')
                sys.exit(1)
            replay_path = matches[-1]
    else:
        # Open file dialog
        replay_path = filedialog.askopenfilename(
            title='Open Halite III Replay',
            filetypes=[('Halite replay', '*.hlt'), ('All files', '*.*')],
        )
        if not replay_path:
            sys.exit(0)

    if not os.path.exists(replay_path):
        print(f'File not found: {replay_path}')
        sys.exit(1)

    print(f'Loading: {replay_path}')
    try:
        app = HaliteViewer(root, replay_path)
    except Exception as e:
        print(f'Error loading replay: {e}')
        import traceback; traceback.print_exc()
        sys.exit(1)

    root.mainloop()


if __name__ == '__main__':
    main()
