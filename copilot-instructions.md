First Rule: Assume the code base is working (clone from a working GitHub repository) and don't make any code change. View the code base as server side implementation which should never be changed.
Second Rule: If there is any code change needed, only do it on the folder 'My Extension', which is the client side implementation, and it uses server side implementation to run the game.

------------------

Game Overview
Game Rules
Halite III is a resource management game in which players build and command ships that explore the ocean and collect halite. Ships use halite as an energy source, and the player with the most stored halite at the end of the game is the winner.

Players begin play with a shipyard, and can use collected halite to build new ships. To efficiently collect halite, players must devise a strategy based on building ships and dropoff points. Ships can explore the ocean, collect halite, and store it in the shipyard or in dropoff points. Players interact by seeking inspiring competition, or by colliding ships to send both to the bottom of the sea.

a centered gif showing wrapping map grid edges and collision

Gameplay Overview
Players each start the game with 5,000 stored halite, a shipyard, and knowledge of the game map. Players play in groups of two or four on a 2d map (32x32, 40x40, 48x48, 56x56, or 64x64) with a unique symmetric pattern of halite.

Ships can make one action per turn: they can move one unit in any cardinal direction, collect halite, or convert into dropoffs. When a ship is over a friendly shipyard or dropoff, it automatically deposits its halite cargo, adding to the player’s total halite.

Ships interact directly in two ways. If multiple ships occupy the same location, they will collide and sink, dropping all their halite into the sea. If there are two or more ships belonging to any opponent within a four-cell radius of your ship, your ship is inspired by the competition. An inspired ship collects halite from the sea at the normal rate, but receives an additional 200% bonus.

Each turn, the game engine sends the players the positions of all ships and dropoffs, and an updated game map. Players have up to two seconds to issue their commands for the turn. The game engine will parse and execute the commands, calculating each player’s resulting halite score and resolving all movement.

Players spend halite to build a ship, move a ship, and to convert a ship to a dropoff. Players gain halite by collecting halite from the sea in their current position.


Command	Halite Cost or Gain
Spawn	Cost: 1000 halite
Convert into a drop-off	Cost: 4000 halite deducted from player’s stored halite.
The converted ship’s halite cargo and the halite in the sea under the new dropoff is credited to the player, potentially reducing the cost.
Move: North, South, East, West	Cost: 10% of halite available at turn origin cell is deducted from ship’s current halite.
When a ship moves over a friendly shipyard or dropoff, it deposits its halite cargo.
Move: Stay still	Gain: 25% of halite available in cell, rounded up to the nearest whole number. Ship remains at its origin.
Ships can carry up to 1000 halite.



Win Conditions
The game continues for 400 to 500 turns, depending on the game map size. The winning player has the most stored halite at the end of the game. If a player does not have a ship or enough energy to construct a ship, they are considered “dead” and lose the game. In the case of a tie, players are ranked by their stored halite on the last turn, then by their stored halite on the penultimate turn, and so on.


---------------------------

Overview
This API Documentation refers to objects shipped with Halite III starter kits:

GAME
PLAYER
SHIP
SHIPYARD
DROPOFF
MAP
MAP CELL
POSITION
DIRECTION


Game
The game object holds all metadata to run the game, and is an organizing layer between your code and the game engine. Game initializes the game, which includes generating the map and registering the players.



Initialization phase

A game of Halite III is initialized when each player sends a string name. Game forwards this to the engine, and launches the game.

game.ready(“name”)



Game loop

The game loop sends the game state to the players and processes commands returned from the players. This repeats for each turn. Games last between 400 and 500 turns per game depending on map size. The game engine kills any bot that takes more than 2,000 milliseconds to process.

game.update_frame() updates the game state, and returns nothing.



Command queue

The command queue is a list of commands. The player’s code fills this list with commands and sends it to the game object, where it is sent to the game engine. The game engine kills any bot that attempts to issue multiple commands to one ship.

game.end_turn([commands])

Valid commands that can be sent to the engine:

Action	Engine Command
GENERATE	g
CONSTRUCT	c
MOVE	m
Move Commands	n, s, e, w and o for origin (stay still)


PLAYER
Players have an id, a shipyard, a halite_amount, and dictionaries of ships and dropoffs as member variables.



Ships

Players can access their ships either singly by id, or all together as a list. See methods on ships below.

player.get_ship(ship_id) returns the ship object associated with the ship id provided as an argument.

player.get_ships() returns a list of all ship objects.

player.has_ship(ship_id) checks if you have a ship with this id.



Dropoffs

Players can access their dropoffs either singly by id, or all together as a list.

player.get_dropoff(dropoff_id) returns the dropoff object associated with the dropoff id provided as an argument.

player.get_dropoffs() returns a list of all dropoff objects.



Access Other Players

Players can access all players’ ships, shipyard, and dropoffs. game.players is a dictionary of player ids keys to player objects in the game.

for player in game.players: loops over each player in the game by player_id key, including you.



SHIP
Ships carry up to 1,000 halite as cargo and can be issued one command per turn via the command queue. Ships automatically deposit their cargo when over the shipyard or dropoff points. If two ships collide, both are destroyed; their cargo falls back into the sea at the collision site.

Ships have an owner, an id, a position, and a halite_amount.



Cargo

ship.is_full returns a boolean True if ship is carrying 1,000 halite (the maximum). Otherwise returns False.



Convert to Dropoff

Ships can be converted into dropoff sites at their present location. The conversion costs 4,000 halite, deducted from total current stored halite. The converted ship’s halite cargo and the halite in the sea under the new dropoff is credited to the player. These credits resolve first, and can be used toward the cost of the dropoff.

ship.make_dropoff() returns an engine command to convert this ship into a dropoff.



Collect Halite at Origin

Ships can collect 25% of the halite from the sea at their present location, rounded up to the nearest integer.

ship.stay_still() returns an engine command to keep this ship where it is and collect halite.



Move

Ships can move one square in a cardinal direction per turn. Each move costs 10% of the halite available in the sea at the ship’s starting location, debited from the ships’ cargo. The direction of the move is communicated via the command queue.

ship.move(direction) returns an engine command to move this ship in a direction without checking for collisions.



SHIPYARD
Each player begins the game with a shipyard. Shipyards have an owner, an id, and a position.



Spawn

shipyard.spawn() returns an engine command to generate a new ship.



DROPOFF
You create a dropoff at any location on the map by converting a ship. Ships can store halite at a dropoff point just as they would at the shipyard. If two dropoffs are constructed in the same location, the engine returns an error and the construct command fails. The player class has the methods to access dropoffs.

Dropoffs have an owner, an id, and a position.



MAP
Gameplay takes place on a wrapping rectangular grid 32x32, 40x40, 48x48, 56x56, or 64x64 in dimension. The map edges wrap to their opposite edge and create a torus shape. The game map can be indexed by a position or by a contained entity (ship, shipyard, or dropoff). The game map has width and height as member variables.



Calculate distance

A method that computes the Manhattan distance between two locations, and accounts for the toroidal wraparound.

game_map.calculate_distance(source, target) returns a number.



Normalize position

A method that normalizes a position within the bounds of the toroidal map. Useful for handling the wraparound modulus arithmetic on x and y. For example, if a ship at (x = 31, y = 4) moves to the east on a 32x32 map, the normalized position would be (x = 0, y = 4), rather than the off-the-map position of (x = 32, y = 4).

game_map.normalize(position) returns a normalized position.



Get Unsafe Moves

A method that returns a list of direction(s) to move closer to a target disregarding collision possibilities. Returns an empty list if the source and destination are the same.

game_map.get_unsafe_moves(source, destination) returns a list of closest directions toward the given target.



Naive Navigate

A method that returns a direction to move closer to a target without colliding with other entities. Returns a direction of “still” if no such move exists.

game_map.naive_navigate(ship, destination) returns a single valid direction toward a given target.



MAP CELL
A map cell is an object representation of a cell on the game map. Map cell has position, halite_amount, ship, and structure as member variables. For example, you can index the game map and find a particular map cell with game_map[position].



Property Accessors

map_cell.is_empty returns True if the cell is empty.

map_cell.is_occupied returns True if there is a ship on this cell.

map_cell.has_structure returns True if there is a structure (a dropoff or shipyard) on this cell.

map_cell.structure_type returns the type of structure on this cell, or None if there is no structure.



Navigational Marking

map_cell.mark_unsafe(ship) is used to mark the cell under this ship as unsafe (occupied) for collision avoidance. This marking resets every turn and is used by naive_navigate to avoid collisions.



POSITION
A position is an object with x and y values indicating the absolute position on the game map. Position is defined in the file hlt/positionals.py. You can use the position information on an entity (entity.position), or create a new position object with Position(x, y).

position.directional_offset(direction) returns a new position based on moving one unit in the given direction from the given position. This method takes a direction such as Direction.West or an equivalent tuple such as (0, -1), but will not work with commands such as "w".

position.get_surrounding_cardinals() returns a list of all positions around the given position in each cardinal direction.



DIRECTION
A direction is a direction of movement: Direction.West, Direction.North, Direction.East, Direction.South. Direction is defined in the file hlt/positionals.py.

Direction.get_all_cardinals() returns an array of all cardinal tuples.

Direction.convert() returns a letter command from a direction tuple.

Direction.invert() returns a letter command of the opposite cardinal direction given a direction tuple.