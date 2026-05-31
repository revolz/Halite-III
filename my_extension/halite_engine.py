#!/usr/bin/env python3
"""
Halite III Python Game Engine

Replicates the C++ game engine for local testing.
Manages bot subprocesses and communicates via stdin/stdout pipes
using the same protocol as the original C++ engine.
"""

import json
import math
import random
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants (mirror of C++ Constants.hpp defaults)
# ---------------------------------------------------------------------------

SHIP_COST = 1000          # NEW_ENTITY_ENERGY_COST
DROPOFF_COST = 4000
MAX_HALITE = 1000         # MAX_ENERGY (ship cargo cap)
INITIAL_ENERGY = 5000
EXTRACT_RATIO = 4         # collect ceil(cell/4) per turn
MOVE_COST_RATIO = 10      # cost = floor(cell/10) from ship cargo
MAX_CELL_PRODUCTION = 1000
MIN_CELL_PRODUCTION = 900
MIN_TURNS = 400
MIN_TURN_THRESHOLD = 32
MAX_TURNS_CAP = 500
MAX_TURN_THRESHOLD = 64
INSPIRATION_ENABLED = True
INSPIRATION_RADIUS = 4
INSPIRATION_SHIP_COUNT = 2
INSPIRED_EXTRACT_RATIO = 4        # same ratio, but bonus applied on top
INSPIRED_BONUS_MULTIPLIER = 2     # +200% bonus halite collected
INSPIRED_MOVE_COST_RATIO = 10     # same as normal (no cost reduction)

CONSTANTS_JSON = {
    "NEW_ENTITY_ENERGY_COST": SHIP_COST,
    "DROPOFF_COST": DROPOFF_COST,
    "MAX_ENERGY": MAX_HALITE,
    "INITIAL_ENERGY": INITIAL_ENERGY,
    "EXTRACT_RATIO": EXTRACT_RATIO,
    "MOVE_COST_RATIO": MOVE_COST_RATIO,
    "INSPIRATION_ENABLED": INSPIRATION_ENABLED,
    "INSPIRATION_RADIUS": INSPIRATION_RADIUS,
    "INSPIRATION_SHIP_COUNT": INSPIRATION_SHIP_COUNT,
    "INSPIRED_EXTRACT_RATIO": INSPIRED_EXTRACT_RATIO,
    "INSPIRED_BONUS_MULTIPLIER": INSPIRED_BONUS_MULTIPLIER,
    "INSPIRED_MOVE_COST_RATIO": INSPIRED_MOVE_COST_RATIO,
    "CAPTURE_ENABLED": False,
    "SHIPS_ABOVE_FOR_CAPTURE": 3,
    "CAPTURE_RADIUS": 3,
    "STRICT_ERRORS": False,
    "MAX_PLAYERS": 16,
    "MIN_TURNS": MIN_TURNS,
    "MAX_TURNS": MAX_TURNS_CAP,
}

# ---------------------------------------------------------------------------
# Replay constants (mirrors C++ Replay.hpp / version.hpp)
# ---------------------------------------------------------------------------

REPLAY_FILE_VERSION = 3
ENGINE_VERSION = "1.5.1239.g3908"

# Direction tuples (dx, dy)
DIRECTIONS = {
    'n': (0, -1),
    's': (0,  1),
    'e': (1,  0),
    'w': (-1, 0),
    'o': (0,  0),  # stay still
}


# ---------------------------------------------------------------------------
# Map Generation: FractalValueNoise + Symmetrical (mirrors C++
# FractalValueNoiseTileGenerator and SymmetricalTile — the default generator)
# ---------------------------------------------------------------------------

# Constants matching C++ Constants.hpp
FACTOR_EXP_1 = 2.0   # power applied to seed noise (biases toward 0)
FACTOR_EXP_2 = 2.0   # power applied after multi-octave blend (makes spikier)
PERSISTENCE  = 0.7   # amplitude multiplier per octave


def _smooth_noise(source: List[List[float]], wavelength: int,
                  tw: int, th: int) -> List[List[float]]:
    """
    Bilinear interpolation of source noise at the given wavelength.
    Mirrors FractalValueNoiseTileGenerator::generateSmoothNoise().
    """
    mini_rows = math.ceil(th / wavelength)
    mini_cols = math.ceil(tw / wavelength)
    mini = [[source[wavelength * y][wavelength * x]
             for x in range(mini_cols)]
            for y in range(mini_rows)]

    out = [[0.0] * tw for _ in range(th)]
    for y in range(th):
        y_i = y // wavelength
        y_f = (y // wavelength + 1) % mini_rows
        v_blend = y / wavelength - y_i
        for x in range(tw):
            x_i = x // wavelength
            x_f = (x // wavelength + 1) % mini_cols
            h_blend = x / wavelength - x_i
            top    = (1 - h_blend) * mini[y_i][x_i] + h_blend * mini[y_i][x_f]
            bottom = (1 - h_blend) * mini[y_f][x_i] + h_blend * mini[y_f][x_f]
            out[y][x] = (1 - v_blend) * top + v_blend * bottom
    return out


def _generate_tile(tw: int, th: int, rng: random.Random) -> List[List[int]]:
    """
    Generate a tw×th tile using fractal value noise.
    Mirrors C++ FractalValueNoiseTileGenerator::generate().

    Produces wide variation from ~0 to max_cell_production, creating the
    characteristic look of the original Halite III maps (small cells in low-
    halite areas, large cells in high-halite hotspots).
    """
    # Seed noise: pow(uniform, FACTOR_EXP_1) biases values toward 0
    source = [[rng.random() ** FACTOR_EXP_1 for _ in range(tw)] for _ in range(th)]

    region = [[0.0] * tw for _ in range(th)]
    max_octave = int(math.floor(math.log2(min(tw, th)))) + 1

    amplitude = 1.0
    for octave in range(2, max_octave + 1):
        wavelength = round(2 ** (max_octave - octave))
        smoothed = _smooth_noise(source, wavelength, tw, th)
        for y in range(th):
            for x in range(tw):
                region[y][x] += amplitude * smoothed[y][x]
        amplitude *= PERSISTENCE

    # Add finest-scale source noise with remaining amplitude
    for y in range(th):
        for x in range(tw):
            region[y][x] += amplitude * source[y][x]

    # Apply spikiness exponent; find max
    max_val = 0.0
    for y in range(th):
        for x in range(tw):
            region[y][x] = region[y][x] ** FACTOR_EXP_2
            if region[y][x] > max_val:
                max_val = region[y][x]

    # Normalize: max cell production is random in [MIN, MAX] (mirrors C++)
    max_prod = rng.randint(MIN_CELL_PRODUCTION, MAX_CELL_PRODUCTION)

    result = [[0] * tw for _ in range(th)]
    if max_val > 0:
        for y in range(th):
            for x in range(tw):
                result[y][x] = round(region[y][x] * max_prod / max_val)
    return result


def _flip_vertical(grid: List[List[int]], curr_w: int, curr_h: int,
                   factories: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Mirror the left curr_w columns to the right (doubles width). Returns new factories."""
    new_factories = []
    for y in range(curr_h):
        for x in range(curr_w):
            grid[y][curr_w * 2 - x - 1] = grid[y][x]
    for (fx, fy) in factories:
        nfx = curr_w * 2 - fx - 1
        grid[fy][nfx] = 0
        new_factories.append((nfx, fy))
    return new_factories


def _flip_horizontal(grid: List[List[int]], curr_w: int, curr_h: int,
                     factories: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Mirror the top curr_h rows downward (doubles height). Returns new factories."""
    new_factories = []
    for y in range(curr_h):
        for x in range(curr_w):
            grid[curr_h * 2 - y - 1][x] = grid[y][x]
    for (fx, fy) in factories:
        nfy = curr_h * 2 - fy - 1
        grid[nfy][fx] = 0
        new_factories.append((fx, nfy))
    return new_factories


def generate_map(width: int, height: int, num_players: int,
                 rng: random.Random) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Generate a symmetric halite map using FractalValueNoise + Symmetrical
    (mirrors C++ FractalValueNoiseTileGenerator + SymmetricalTile, the default).

    Returns (grid[y][x], factories[(x,y), ...]) where factories has exactly
    num_players entries in player-id order.
    """
    # Determine tile dimensions by doubling (mirrors SymmetricalTile constructor)
    num_tiles = 1
    num_tile_cols = 1
    num_tile_rows = 1
    while num_tiles < num_players:
        num_tile_cols *= 2
        num_tiles *= 2
        if num_tiles == num_players:
            break
        num_tile_rows *= 2
        num_tiles *= 2

    tw = width  // num_tile_cols
    th = height // num_tile_rows

    tile = _generate_tile(tw, th, rng)

    # Full grid (pre-filled to 0)
    grid = [[0] * width for _ in range(height)]
    for y in range(th):
        for x in range(tw):
            grid[y][x] = tile[y][x]

    # Factory position inside tile (mirrors SymmetricalTile::tile_map)
    if 16 <= tw <= 40 and 16 <= th <= 40:
        fx0 = int(8 + ((tw - 16) / 24.0) * 20)
        fy0 = int(8 + ((th - 16) / 24.0) * 20) if num_players > 2 else th // 2
    else:
        fx0 = tw // 2
        fy0 = th // 2

    grid[fy0][fx0] = 0
    all_factories = [(fx0, fy0)]

    # Expand via flip operations
    num_tiles = 1
    curr_w, curr_h = tw, th
    while num_tiles < num_players:
        new_facts = _flip_vertical(grid, curr_w, curr_h, all_factories)
        all_factories.extend(new_facts)
        curr_w *= 2
        num_tiles *= 2
        if num_tiles == num_players:
            break
        new_facts = _flip_horizontal(grid, curr_w, curr_h, all_factories)
        all_factories.extend(new_facts)
        curr_h *= 2
        num_tiles *= 2

    return grid, all_factories[:num_players]


# ---------------------------------------------------------------------------
# Halite Engine
# ---------------------------------------------------------------------------

class HaliteEngine:
    """
    Python reimplementation of the Halite III C++ game engine.

    State containers:
      halite[(x,y)]              – halite on that cell
      cell_owner[(x,y)]          – player_id owning the cell (factory/dropoff), or None
      cell_entity[(x,y)]         – ship entity_id on the cell, or None
      entities[eid]              – {'owner': pid, 'cargo': int, 'is_inspired': bool}
      player_entities[pid]       – {eid: (x, y)}  (canonical position store)
      players[pid]               – {'energy': int, 'factory': (x,y),
                                    'dropoffs': [(did, x, y)]}
    """

    def __init__(self, width: int, height: int, num_players: int,
                 seed: Optional[int] = None, verbose: bool = False):
        self.width = width
        self.height = height
        self.num_players = num_players
        self.verbose = verbose

        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.rng = random.Random(self.seed)

        # Compute max turns (mirrors HaliteImpl::initialize_game)
        max_dim = max(width, height)
        turns = MIN_TURNS
        if max_dim > MIN_TURN_THRESHOLD:
            turns += int(
                ((max_dim - MIN_TURN_THRESHOLD) /
                 (MAX_TURN_THRESHOLD - MIN_TURN_THRESHOLD))
                * (MAX_TURNS_CAP - MIN_TURNS)
            )
        self.max_turns = turns

        # Map state
        self.halite: Dict[Tuple[int, int], int] = {}
        self.cell_owner: Dict[Tuple[int, int], int] = {}   # pid
        self.cell_entity: Dict[Tuple[int, int], Optional[int]] = {}

        # Entity state
        self._next_ship_id = 0
        self._next_dropoff_id = 0
        self.entities: Dict[int, dict] = {}
        self.player_entities: Dict[int, Dict[int, Tuple[int, int]]] = {}

        # Player state
        self.players: Dict[int, dict] = {}

        # Per-turn tracking
        self.turn = 0
        self.changed_cells: Set[Tuple[int, int]] = set()
        self._moved_entities: Set[int] = set()  # entities that moved/deposited/spawned

        # Replay tracking (populated during run())
        self._replay_frames: List[dict] = []
        self._current_events: List[dict] = []  # events for the current turn

        # Saved initial state for production_map in replay
        self._initial_grid: List[List[int]] = []
        self._initial_factories: List[Tuple[int, int]] = []
        self._initial_total_halite: int = 0

        # Per-player statistics (initialized in _init_map_and_players)
        self._total_deposited: Dict[int, int] = {}
        self._ships_spawned: Dict[int, int] = {}
        self._ships_peak: Dict[int, int] = {}
        self._last_turn_alive: Dict[int, int] = {}
        self._last_turn_ship_spawn: Dict[int, int] = {}
        self._all_collisions: Dict[int, int] = {}
        self._total_dropped: Dict[int, int] = {}
        self._total_mined: Dict[int, int] = {}
        self._total_bonus: Dict[int, int] = {}
        self._number_dropoffs: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _norm(self, x: int, y: int) -> Tuple[int, int]:
        return x % self.width, y % self.height

    def _dist(self, x1: int, y1: int, x2: int, y2: int) -> int:
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return min(dx, self.width - dx) + min(dy, self.height - dy)

    def _new_ship_id(self) -> int:
        eid = self._next_ship_id
        self._next_ship_id += 1
        return eid

    def _new_dropoff_id(self) -> int:
        did = self._next_dropoff_id
        self._next_dropoff_id += 1
        return did

    def _dump_halite(self, pos: Tuple[int, int], amount: int):
        """Dump `amount` halite: credits cell owner if owned, else adds to cell."""
        if amount <= 0:
            return
        owner = self.cell_owner.get(pos)
        if owner is not None:
            self.players[owner]['energy'] += amount
        else:
            self.halite[pos] = self.halite.get(pos, 0) + amount
            self.changed_cells.add(pos)

    # ------------------------------------------------------------------
    # Map + game initialisation
    # ------------------------------------------------------------------

    def _init_map_and_players(self):
        grid, factories = generate_map(self.width, self.height, self.num_players, self.rng)

        # Save initial state for replay
        self._initial_grid = [row[:] for row in grid]
        self._initial_factories = list(factories)
        self._initial_total_halite = sum(
            grid[y][x] for y in range(self.height) for x in range(self.width)
        )

        for y in range(self.height):
            for x in range(self.width):
                self.halite[(x, y)] = grid[y][x]

        for pid, (fx, fy) in enumerate(factories):
            self.players[pid] = {
                'energy': INITIAL_ENERGY,
                'factory': (fx, fy),
                'dropoffs': [],
            }
            self.player_entities[pid] = {}
            self.cell_owner[(fx, fy)] = pid
            self.halite[(fx, fy)] = 0  # factories always have 0 halite

            # Initialize per-player statistics
            self._total_deposited[pid] = 0
            self._ships_spawned[pid] = 0
            self._ships_peak[pid] = 0
            self._last_turn_alive[pid] = 0
            self._last_turn_ship_spawn[pid] = 0
            self._all_collisions[pid] = 0
            self._total_dropped[pid] = 0
            self._total_mined[pid] = 0
            self._total_bonus[pid] = 0
            self._number_dropoffs[pid] = 0

    # ------------------------------------------------------------------
    # Inspiration
    # ------------------------------------------------------------------

    def _update_inspiration(self):
        if not INSPIRATION_ENABLED:
            return
        for pid, p_ents in self.player_entities.items():
            for eid, (ex, ey) in p_ents.items():
                opponent_ships = 0
                for dx in range(-INSPIRATION_RADIUS, INSPIRATION_RADIUS + 1):
                    for dy in range(-INSPIRATION_RADIUS, INSPIRATION_RADIUS + 1):
                        nx, ny = self._norm(ex + dx, ey + dy)
                        if self._dist(ex, ey, nx, ny) > INSPIRATION_RADIUS:
                            continue
                        near_eid = self.cell_entity.get((nx, ny))
                        if near_eid is not None and self.entities[near_eid]['owner'] != pid:
                            opponent_ships += 1
                self.entities[eid]['is_inspired'] = (
                    opponent_ships >= INSPIRATION_SHIP_COUNT
                )

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _capture_entities_for_replay(self) -> dict:
        """Snapshot all entity positions/cargo/inspiration at the START of a turn."""
        result: dict = {}
        for pid in self.players:
            result[str(pid)] = {}
            for eid, (ex, ey) in self.player_entities[pid].items():
                result[str(pid)][str(eid)] = {
                    'x': ex, 'y': ey,
                    'energy': self.entities[eid]['cargo'],
                    'is_inspired': self.entities[eid]['is_inspired'],
                }
        return result

    def _commands_to_moves_json(self, all_commands: Dict[int, str]) -> dict:
        """Convert raw bot command strings to the move-list format used in .hlt files."""
        moves: dict = {}
        for pid, cmd_str in all_commands.items():
            parsed = self._parse_commands(cmd_str, pid)
            player_moves = []
            for cmd in parsed:
                if cmd[0] == 'spawn':
                    player_moves.append({'type': 'g', 'energy': 0})
                elif cmd[0] == 'move':
                    _, ship_id, direction = cmd
                    if direction != 'o':
                        player_moves.append({'type': 'm', 'id': ship_id, 'direction': direction})
                elif cmd[0] == 'construct':
                    _, ship_id = cmd
                    player_moves.append({'type': 'c', 'id': ship_id})
            moves[str(pid)] = player_moves
        return moves

    def _build_game_constants(self) -> dict:
        """Build the GAME_CONSTANTS dict matching C++ Constants.cpp serialization."""
        return {
            "STRICT_ERRORS": False,
            "MAX_PLAYERS": 16,
            "DEFAULT_MAP_WIDTH": self.width,
            "DEFAULT_MAP_HEIGHT": self.height,
            "MAX_CELL_PRODUCTION": MAX_CELL_PRODUCTION,
            "MIN_CELL_PRODUCTION": MIN_CELL_PRODUCTION,
            "MAX_ENERGY": MAX_HALITE,
            "NEW_ENTITY_ENERGY_COST": SHIP_COST,
            "INITIAL_ENERGY": INITIAL_ENERGY,
            "DROPOFF_COST": DROPOFF_COST,
            "MOVE_COST_RATIO": MOVE_COST_RATIO,
            "DROPOFF_PENALTY_RATIO": 4,
            "EXTRACT_RATIO": EXTRACT_RATIO,
            "PERSISTENCE": 0.7,
            "FACTOR_EXP_1": 2.0,
            "FACTOR_EXP_2": 2.0,
            "MIN_TURNS": MIN_TURNS,
            "MIN_TURN_THRESHOLD": MIN_TURN_THRESHOLD,
            "MAX_TURNS": self.max_turns,
            "MAX_TURN_THRESHOLD": MAX_TURN_THRESHOLD,
            "CAPTURE_ENABLED": False,
            "CAPTURE_RADIUS": 3,
            "SHIPS_ABOVE_FOR_CAPTURE": 3,
            "INSPIRATION_ENABLED": INSPIRATION_ENABLED,
            "INSPIRED_EXTRACT_RATIO": INSPIRED_EXTRACT_RATIO,
            "INSPIRED_BONUS_MULTIPLIER": float(INSPIRED_BONUS_MULTIPLIER),
            "INSPIRED_MOVE_COST_RATIO": INSPIRED_MOVE_COST_RATIO,
            "INSPIRATION_RADIUS": INSPIRATION_RADIUS,
            "INSPIRATION_SHIP_COUNT": INSPIRATION_SHIP_COUNT,
        }

    def _build_replay(self, player_names: List[str]) -> dict:
        """Build the full replay dict matching the .hlt JSON format."""
        # production_map: grid[y][x] = {"energy": halite}
        production_map = {
            'width': self.width,
            'height': self.height,
            'grid': [
                [{'energy': self._initial_grid[y][x]} for x in range(self.width)]
                for y in range(self.height)
            ],
        }

        # Players: initial state (energy=INITIAL_ENERGY, no entities yet)
        players_json = []
        for pid in range(self.num_players):
            fx, fy = self._initial_factories[pid]
            players_json.append({
                'player_id': pid,
                'name': player_names[pid],
                'energy': INITIAL_ENERGY,
                'factory_location': {'x': fx, 'y': fy},
                'entities': [],
            })

        # Rank players by final energy (descending)
        ranked_pids = sorted(range(self.num_players),
                             key=lambda pid: -self.players[pid]['energy'])
        player_stats = []
        for rank, pid in enumerate(ranked_pids, 1):
            lifespan = sum(
                1 for _ in self.player_entities[pid]
            ) if self.player_entities[pid] else 0
            player_stats.append({
                'player_id': pid,
                'rank': rank,
                'last_turn_alive': self._last_turn_alive.get(pid, self.turn),
                'last_turn_ship_spawn': self._last_turn_ship_spawn.get(pid, 0),
                'final_production': self.players[pid]['energy'],
                'total_production': self._total_deposited.get(pid, 0),
                'max_entity_distance': 0,
                'number_dropoffs': self._number_dropoffs.get(pid, 0),
                'interaction_opportunities': 0,
                'ships_spawned': self._ships_spawned.get(pid, 0),
                'ships_peak': self._ships_peak.get(pid, 0),
                'self_collisions': 0,
                'all_collisions': self._all_collisions.get(pid, 0),
                'dropoff_collisions': 0,
                'total_mined': self._total_mined.get(pid, 0),
                'total_bonus': self._total_bonus.get(pid, 0),
                'total_dropped': self._total_dropped.get(pid, 0),
                'carried_at_end': sum(
                    e['cargo'] for e in self.entities.values() if e['owner'] == pid
                ),
                'mining_efficiency': 0.0,
                'average_entity_distance': 0,
                'halite_per_dropoff': {},
                'random_id': 0,
            })

        game_statistics = {
            'number_turns': self.turn,
            'execution_time': 0,
            'map_total_halite': self._initial_total_halite,
            'player_statistics': player_stats,
        }

        # Frame 0: empty initial frame (before any turns)
        frame_0 = {'cells': [], 'energy': {}, 'entities': {}, 'events': [], 'moves': {}}

        return {
            'ENGINE_VERSION': ENGINE_VERSION,
            'GAME_CONSTANTS': self._build_game_constants(),
            'REPLAY_FILE_VERSION': REPLAY_FILE_VERSION,
            'number_of_players': self.num_players,
            'map_generator_seed': self.seed,
            'production_map': production_map,
            'players': players_json,
            'game_statistics': game_statistics,
            'full_frames': [frame_0] + self._replay_frames,
        }

    def write_replay(self, filename: str, player_names: List[str]):
        """Serialize the replay to a .hlt file (zstd-compressed JSON)."""
        import os as _os
        replay_data = self._build_replay(player_names)
        json_bytes = json.dumps(replay_data).encode('utf-8')

        _os.makedirs(_os.path.dirname(_os.path.abspath(filename)), exist_ok=True)

        try:
            import zstd
            data = zstd.compress(json_bytes)
        except ImportError:
            # Fall back to plain JSON if zstd not installed
            data = json_bytes

        with open(filename, 'wb') as f:
            f.write(data)

    # ------------------------------------------------------------------
    # Command processing (transaction order: construct → move → dump → spawn)
    # ------------------------------------------------------------------

    def _parse_commands(self, cmd_str: str, pid: int) -> List[tuple]:
        """Parse a command string into a list of typed command tuples."""
        tokens = cmd_str.split()
        cmds = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == 'g':
                cmds.append(('spawn',))
                i += 1
            elif t == 'm' and i + 2 < len(tokens):
                try:
                    ship_id = int(tokens[i + 1])
                    direction = tokens[i + 2]
                    cmds.append(('move', ship_id, direction))
                except ValueError:
                    pass
                i += 3
            elif t == 'c' and i + 1 < len(tokens):
                try:
                    ship_id = int(tokens[i + 1])
                    cmds.append(('construct', ship_id))
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        return cmds

    def _process_commands(self, all_commands: Dict[int, str]):
        """
        Process all player commands for one turn.
        Transaction order (mirrors C++ CommandTransaction all_transactions):
          construct → move → dump → spawn
        Then mining runs separately in _process_mining().
        """
        parsed = {
            pid: self._parse_commands(cmd_str, pid)
            for pid, cmd_str in all_commands.items()
        }

        # Snapshot player energies for pre-deposit affordability checks
        pre_energy = {pid: p['energy'] for pid, p in self.players.items()}
        # Track cumulative expenses per player for validation
        expenses = {pid: 0 for pid in self.players}

        # ── 1. Construct ────────────────────────────────────────────────
        commanded_ships: Set[int] = set()  # ships that have received a command

        for pid, cmds in parsed.items():
            for cmd in cmds:
                if cmd[0] != 'construct':
                    continue
                ship_id = cmd[1]

                # Validate: ship exists and belongs to player
                if ship_id not in self.player_entities[pid]:
                    continue
                # One command per ship
                if ship_id in commanded_ships:
                    continue

                ex, ey = self.player_entities[pid][ship_id]

                # Cell must not already be owned
                if self.cell_owner.get((ex, ey)) is not None:
                    continue

                cell_h = self.halite.get((ex, ey), 0)
                cargo  = self.entities[ship_id]['cargo']
                credit = cell_h + cargo
                cost   = max(0, DROPOFF_COST - credit)

                # Validate affordability against pre-deposit energy
                if expenses[pid] + cost > pre_energy[pid]:
                    continue

                expenses[pid] += cost
                commanded_ships.add(ship_id)

                # Commit construct -----------------------------------------
                did = self._new_dropoff_id()
                self.cell_owner[(ex, ey)] = pid
                self.players[pid]['dropoffs'].append((did, ex, ey))

                # Credit cell + cargo to player (dump_energy to now-owned cell)
                self.players[pid]['energy'] += credit
                # Deduct construction cost
                self.players[pid]['energy'] -= cost

                self.halite[(ex, ey)] = 0
                self.changed_cells.add((ex, ey))
                self.cell_entity[(ex, ey)] = None

                del self.entities[ship_id]
                del self.player_entities[pid][ship_id]

                # Replay event + stats
                self._current_events.append({
                    'type': 'construct',
                    'location': {'x': ex, 'y': ey},
                    'owner_id': pid,
                    'id': ship_id,
                })
                self._number_dropoffs[pid] += 1

        # ── 2. Move ─────────────────────────────────────────────────────
        # Phase 2a: lift movers off their source cells
        # destinations[(x,y)] = [eid, ...]  of ships heading there
        destinations: Dict[Tuple[int, int], List[int]] = {}
        # Track which entities are "in transit" (lifted off the grid)
        in_transit: Set[int] = set()
        # Map eid → destination for movers
        mover_dest: Dict[int, Tuple[int, int]] = {}

        for pid, cmds in parsed.items():
            for cmd in cmds:
                if cmd[0] != 'move':
                    continue
                ship_id, dir_str = cmd[1], cmd[2]

                if ship_id not in self.player_entities[pid]:
                    continue
                if ship_id in commanded_ships:
                    continue
                # Count commands per entity: first move command wins
                commanded_ships.add(ship_id)

                direction = DIRECTIONS.get(dir_str, (0, 0))
                if direction == (0, 0):
                    # Stay-still is a no-op (ship can still mine)
                    commanded_ships.discard(ship_id)
                    continue

                ex, ey = self.player_entities[pid][ship_id]
                dx, dy = direction
                dest = self._norm(ex + dx, ey + dy)

                # Check move affordability (deducted from ship cargo)
                cell_h = self.halite.get((ex, ey), 0)
                move_cost = cell_h // MOVE_COST_RATIO
                if self.entities[ship_id]['cargo'] < move_cost:
                    # Insufficient cargo: silently ignore, ship can mine
                    commanded_ships.discard(ship_id)
                    continue

                # Deduct move cost from ship cargo
                self.entities[ship_id]['cargo'] -= move_cost

                # Lift ship off source cell
                if self.cell_entity.get((ex, ey)) == ship_id:
                    self.cell_entity[(ex, ey)] = None
                del self.player_entities[pid][ship_id]
                in_transit.add(ship_id)
                mover_dest[ship_id] = dest

                destinations.setdefault(dest, []).append(ship_id)

        # Phase 2b: also lift stationary ships sitting at contested destinations
        for dest in list(destinations.keys()):
            existing = self.cell_entity.get(dest)
            if existing is not None and existing not in in_transit:
                owner = self.entities[existing]['owner']
                del self.player_entities[owner][existing]
                self.cell_entity[dest] = None
                in_transit.add(existing)
                mover_dest[existing] = dest
                destinations[dest].append(existing)

        # Phase 2c: resolve destinations
        for dest, arrivals in destinations.items():
            if len(arrivals) > 1:
                # Collect stats and emit collision event BEFORE deleting entities
                self._current_events.append({
                    'type': 'shipwreck',
                    'location': {'x': dest[0], 'y': dest[1]},
                    'ships': [{'id': eid, 'owner': self.entities[eid]['owner']}
                              for eid in arrivals],
                })
                for eid in arrivals:
                    owner = self.entities[eid]['owner']
                    self._all_collisions[owner] += 1
                    self._total_dropped[owner] += self.entities[eid]['cargo']
                    self._dump_halite(dest, self.entities[eid]['cargo'])
                    del self.entities[eid]
                self.cell_entity[dest] = None
                self.changed_cells.add(dest)
            else:
                eid = arrivals[0]
                owner = self.entities[eid]['owner']
                self.cell_entity[dest] = eid
                self.player_entities[owner][eid] = dest
                self._moved_entities.add(eid)
                self.changed_cells.add(dest)

        # ── 3. Dump (auto-deposit) ───────────────────────────────────────
        for pid, p_ents in self.player_entities.items():
            for eid, (ex, ey) in list(p_ents.items()):
                if self.cell_owner.get((ex, ey)) == pid:
                    cargo = self.entities[eid]['cargo']
                    if cargo > 0:
                        self.players[pid]['energy'] += cargo
                        self._total_deposited[pid] += cargo
                        self.entities[eid]['cargo'] = 0
                    self._moved_entities.add(eid)  # deposited → can't mine

        # ── 4. Spawn ────────────────────────────────────────────────────
        for pid, cmds in parsed.items():
            spawn_requested = any(c[0] == 'spawn' for c in cmds)
            if not spawn_requested:
                continue
            # Validate affordability against pre-deposit energy (expenses already
            # accumulated from constructs this turn)
            if expenses[pid] + SHIP_COST > pre_energy[pid]:
                continue

            fx, fy = self.players[pid]['factory']
            self.players[pid]['energy'] -= SHIP_COST

            new_eid = self._new_ship_id()
            self.entities[new_eid] = {'owner': pid, 'cargo': 0, 'is_inspired': False}

            existing = self.cell_entity.get((fx, fy))
            if existing is None:
                self.cell_entity[(fx, fy)] = new_eid
                self.player_entities[pid][new_eid] = (fx, fy)
                self._moved_entities.add(new_eid)
                # Replay: spawn event + stats
                self._current_events.append({
                    'type': 'spawn',
                    'location': {'x': fx, 'y': fy},
                    'owner_id': pid,
                    'id': new_eid,
                    'energy': 0,
                })
                self._ships_spawned[pid] += 1
                self._last_turn_ship_spawn[pid] = self.turn
            else:
                # Collision at factory: dump existing ship's cargo to factory owner,
                # destroy both ships
                existing_cargo = self.entities[existing]['cargo']
                self.players[pid]['energy'] += existing_cargo
                existing_owner = self.entities[existing]['owner']
                # Replay: collision event + stats
                self._current_events.append({
                    'type': 'shipwreck',
                    'location': {'x': fx, 'y': fy},
                    'ships': [{'id': new_eid, 'owner': pid},
                              {'id': existing, 'owner': existing_owner}],
                })
                self._all_collisions[pid] += 1
                self._all_collisions[existing_owner] += 1
                self._total_dropped[existing_owner] += existing_cargo
                if new_eid in self.player_entities.get(pid, {}):
                    del self.player_entities[pid][new_eid]
                if existing in self.player_entities.get(existing_owner, {}):
                    del self.player_entities[existing_owner][existing]
                del self.entities[existing]
                del self.entities[new_eid]
                self.cell_entity[(fx, fy)] = None
            self.changed_cells.add((fx, fy))

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def _process_mining(self):
        """Ships that did not move/deposit/spawn this turn collect halite."""
        for pid, p_ents in self.player_entities.items():
            for eid, (ex, ey) in list(p_ents.items()):
                if eid in self._moved_entities:
                    continue
                if self.entities[eid]['cargo'] >= MAX_HALITE:
                    continue

                cell_h = self.halite.get((ex, ey), 0)
                if cell_h == 0:
                    continue

                is_inspired = self.entities[eid]['is_inspired']
                ratio = INSPIRED_EXTRACT_RATIO if is_inspired else EXTRACT_RATIO

                extracted = math.ceil(cell_h / ratio)
                if extracted == 0 and cell_h > 0:
                    extracted = cell_h

                # Cap extraction so ship doesn't overflow
                cap = MAX_HALITE - self.entities[eid]['cargo']
                if extracted > cap:
                    extracted = cap

                # Apply inspiration bonus on the gained amount
                gained = extracted
                if is_inspired and INSPIRED_BONUS_MULTIPLIER > 0:
                    gained = extracted + INSPIRED_BONUS_MULTIPLIER * extracted
                if gained > cap:
                    gained = cap

                self.entities[eid]['cargo'] += gained
                self.halite[(ex, ey)] = cell_h - extracted
                self.changed_cells.add((ex, ey))

                # Replay stats
                pid = self.entities[eid]['owner']
                self._total_mined[pid] += extracted
                if is_inspired and gained > extracted:
                    self._total_bonus[pid] += gained - extracted

    # ------------------------------------------------------------------
    # Game-end check
    # ------------------------------------------------------------------

    def _game_ended(self) -> bool:
        # All map halite exhausted and all ships empty
        total_map = sum(self.halite.values())
        if total_map == 0:
            if all(e['cargo'] == 0 for e in self.entities.values()):
                return True

        # Count players that can still act
        alive = 0
        for pid, player in self.players.items():
            has_ships = bool(self.player_entities[pid])
            can_spawn = player['energy'] >= SHIP_COST
            if has_ships or can_spawn:
                alive += 1

        if alive > 1:
            return False
        # Single-player game: let them keep playing until turn limit
        return not (self.num_players == 1 and alive == 1)

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------

    def _init_message(self, pid: int) -> str:
        c = dict(CONSTANTS_JSON)
        c['map_width'] = self.width
        c['map_height'] = self.height
        c['game_seed'] = self.seed
        lines = [json.dumps(c)]
        lines.append(f"{self.num_players} {pid}")
        for p in range(self.num_players):
            fx, fy = self.players[p]['factory']
            lines.append(f"{p} {fx} {fy}")
        lines.append(f"{self.width} {self.height}")
        for y in range(self.height):
            lines.append(" ".join(str(self.halite.get((x, y), 0))
                                  for x in range(self.width)))
        return "\n".join(lines) + "\n"

    def _turn_message(self) -> str:
        lines = [str(self.turn)]
        for pid in range(self.num_players):
            player = self.players[pid]
            ships    = self.player_entities[pid]
            dropoffs = player['dropoffs']
            lines.append(f"{pid} {len(ships)} {len(dropoffs)} {player['energy']}")
            for eid, (sx, sy) in ships.items():
                lines.append(f"{eid} {sx} {sy} {self.entities[eid]['cargo']}")
            for did, dx, dy in dropoffs:
                lines.append(f"{did} {dx} {dy}")
        lines.append(str(len(self.changed_cells)))
        for cx, cy in self.changed_cells:
            lines.append(f"{cx} {cy} {self.halite.get((cx, cy), 0)}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bot_cmds: List[str], replay_file: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Run a complete game with the given bot command strings.

        Args:
            bot_cmds: Shell commands to launch each bot.
            replay_file: Optional path for the .hlt replay file. If None, no
                         replay is written. If the path ends with '.hlt', zstd
                         compression is used (requires the ``zstd`` package).

        Returns a ranked list of (player_id, final_halite) tuples.
        """
        assert len(bot_cmds) == self.num_players, \
            f"Expected {self.num_players} bot commands, got {len(bot_cmds)}"

        self._init_map_and_players()

        # Launch bot subprocesses
        env_override = {'PYTHONUNBUFFERED': '1'}
        import os
        env = {**os.environ, **env_override}

        bots = []
        for cmd in bot_cmds:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
            bots.append(proc)

        # Initialization handshake
        player_names = []
        for pid, bot in enumerate(bots):
            bot.stdin.write(self._init_message(pid))
            bot.stdin.flush()
            name = bot.stdout.readline().strip()
            player_names.append(name)
            if self.verbose:
                print(f"  Player {pid}: {name}")

        if self.verbose:
            print(f"Game started: {self.num_players}p  {self.width}x{self.height}  "
                  f"max_turns={self.max_turns}  seed={self.seed}")

        # Game loop
        for self.turn in range(1, self.max_turns + 1):
            # Inspiration is updated before sending the frame (mirrors C++)
            self._update_inspiration()

            # Capture entity state at START of turn (before commands) for replay
            frame_entities = self._capture_entities_for_replay()

            # Build and send turn state (uses changed_cells from previous turn)
            turn_msg = self._turn_message()
            for bot in bots:
                bot.stdin.write(turn_msg)
                bot.stdin.flush()

            # Collect commands
            all_commands: Dict[int, str] = {}
            for pid, bot in enumerate(bots):
                line = bot.stdout.readline()
                all_commands[pid] = line.strip() if line else ''

            # Reset per-turn tracking
            self._current_events = []
            self.changed_cells.clear()
            self._moved_entities.clear()
            self._process_commands(all_commands)
            self._process_mining()

            # Update per-player statistics
            for pid in self.players:
                if self.player_entities[pid] or self.players[pid]['energy'] >= SHIP_COST:
                    self._last_turn_alive[pid] = self.turn
                num_ships = len(self.player_entities[pid])
                if num_ships > self._ships_peak[pid]:
                    self._ships_peak[pid] = num_ships

            # Append the completed frame to the replay
            self._replay_frames.append({
                'entities': frame_entities,
                'moves': self._commands_to_moves_json(all_commands),
                'events': list(self._current_events),
                'cells': [
                    {'x': cx, 'y': cy, 'production': self.halite.get((cx, cy), 0)}
                    for cx, cy in sorted(self.changed_cells)
                ],
                'energy': {str(pid): self.players[pid]['energy'] for pid in self.players},
            })

            if self.verbose and self.turn % 50 == 0:
                scores = {pid: self.players[pid]['energy'] for pid in self.players}
                ships  = {pid: len(self.player_entities[pid]) for pid in self.players}
                print(f"  Turn {self.turn:3d}: {scores}  ships={ships}")

            if self._game_ended():
                break

        # Final results
        results = sorted(
            [(pid, self.players[pid]['energy']) for pid in self.players],
            key=lambda x: -x[1]
        )

        # Write replay file if requested
        if replay_file:
            self.write_replay(replay_file, player_names)
            if self.verbose:
                print(f"  Replay written: {replay_file}")

        # Cleanup
        for bot in bots:
            try:
                bot.stdin.close()
                bot.terminate()
                bot.wait(timeout=2)
            except Exception:
                pass

        return results
