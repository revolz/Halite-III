#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

# Import my libraries
from hlt.positionals import Position

# My Imports
import numpy as np


""" <<<Useful Library>>> """

def min_cost_path_whole(cost):
    # Minimum Cost Path for the whole Cost Array means from the first point [0,0] to the last point [row-1,col-1]
    return min_cost_path(cost, cost.shape[0]-1, cost.shape[1]-1)

# min_cost_path function from https://www.geeksforgeeks.org/min-cost-path-dp-6/
def min_cost_path(cost, m, n): 
    # Take note of the convention expection of X-Y vs Row-Col
    tc = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    tc[0][0] = cost[0][0] 
  
    # Initialize first column of total cost(tc) array 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
  
    # Initialize first row of tc array 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
  
    # Construct rest of the tc array 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j] 

    return tc[m][n], np.array(tc)


def min_cost_path_step(mcp_array):
    # To simplify the algorithm, adding an additional first row and first column with 'ridiculously big cost'
    mcp_array2 = np.ones((mcp_array.shape[0]+1, mcp_array.shape[1]+1)) * 9999
    
    # The cost array will take up the place, except the first row and first column
    mcp_array2[1:,1:] = mcp_array

    # Build the step array
    step = np.array([])

    # The first step is always the target (trace back approach is used this version)
    r=mcp_array2.shape[0]-1
    c=mcp_array2.shape[1]-1

    # As long as we haven't reached the origin cell, trace the lower cost path
    while(r!=1 or c!=1):
        step = np.append(step, (r,c))
        if(mcp_array2[r-1,c] < mcp_array2[r,c-1] ):
            r=r-1
        else:
            c=c-1
    
    # Python doesn't have do-while loop, this line is to include the (1,1), the origin cell in the data structure
    step = np.append(step, (r,c))

    # Reshape to the right array dimension
    step = np.reshape(step,(-1,2))
    
    # return step
    return np.diff(step, axis=0).astype(int)


def center_position(map):
    return np.array(((map.shape[0]-1)/2, (map.shape[1]-1)/2)).astype(int)


def quarter_relative_to_origin( origin_position, target_position ):
    delta = target_position - origin_position
    
    if( delta[0] >= 0 and delta[1] >= 0 ):
        return 4

    elif( delta[0] <= 0 and delta[1] <= 0 ):
        return 1

    elif( delta[0] < 0 and delta[1] > 0 ):
        return 2

    elif( delta[0] > 0 and delta[1] < 0 ):
        return 3

    else:
        return -1


def min_cost_path_to_a_cell(cost_map, target_position):
    origin_position = center_position(cost_map)
    quarter = quarter_relative_to_origin(origin_position, target_position)

    # Divide the map into 4 quarters, in order to fit to the MCP function and speed things up
    if( quarter == 4 ): # Lower Right
        # Cut the right section of map subset according to the quarter
        cost_map_subset = cost_map[origin_position[0]:,origin_position[1]:]
        # No need to flip for Quarter 4

        # Calculate mcps (minimum cost path sum) and mcp (minimum cost path)
        mcps, mcp = min_cost_path(cost_map_subset, 1*(target_position[0]-origin_position[0]), 1*(target_position[1]-origin_position[1]))
        # Since the min_cost_path_step() is using reverse-tracing approach, we multiply with the reverse direction to get from origin to target
        mcp_step = min_cost_path_step(mcp) * [-1, -1]

    elif( quarter == 1 ): # Upper Left
        # Cut the right section of map subset according to the quarter
        cost_map_subset = cost_map[:origin_position[0]+1,:origin_position[1]+1]
        # Flip to the right orientation according to the quarter
        cost_map_subset = np.flip(np.flip(cost_map_subset, axis=0), axis=1)
        
        # Calculate mcps (minimum cost path sum) and mcp (minimum cost path)
        mcps, mcp = min_cost_path(cost_map_subset, -1*(target_position[0]-origin_position[0]), -1*(target_position[1]-origin_position[1]))
        # Since the min_cost_path_step() is using reverse-tracing approach, we multiply with the reverse direction to get from origin to target
        mcp_step = min_cost_path_step(mcp) * [1, 1]

    elif( quarter == 2 ): # Upper Right
        # Cut the right section of map subset according to the quarter
        cost_map_subset = cost_map[:origin_position[0]+1,origin_position[1]:]
        # Flip to the right orientation according to the quarter
        cost_map_subset = np.flip(cost_map_subset, axis=0)
        
        # Calculate mcps (minimum cost path sum) and mcp (minimum cost path)
        mcps, mcp = min_cost_path(cost_map_subset, -1*(target_position[0]-origin_position[0]), 1*(target_position[1]-origin_position[1]))
        # Since the min_cost_path_step() is using reverse-tracing approach, we multiply with the reverse direction to get from origin to target
        mcp_step = min_cost_path_step(mcp) * [1, -1]

    elif( quarter == 3 ): # Lower Left
        # Cut the right section of map subset according to the quarter
        cost_map_subset = cost_map[origin_position[0]:,:origin_position[1]+1]
        # Flip to the right orientation according to the quarter
        cost_map_subset = np.flip(cost_map_subset, axis=1)
        
        # Calculate mcps (minimum cost path sum) and mcp (minimum cost path)
        mcps, mcp = min_cost_path(cost_map_subset, 1*(target_position[0]-origin_position[0]), -1*(target_position[1]-origin_position[1]))
        # Since the min_cost_path_step() is using reverse-tracing approach, we multiply with the reverse direction to get from origin to target
        mcp_step = min_cost_path_step(mcp) * [-1, 1]

    # Take note, the mcp_step is last one to execute first to allow stack like operation for First In Last One (FILO)
    return mcps, mcp, mcp_step.tolist()


def calculate_manhattan_distance(origin, target):
    return abs(origin[0]-target[0]) + abs(origin[1]-target[1]) 

def create_roi_map(halite_map):
    origin_position = center_position(halite_map)

    total = len(halite_map)

    d_array = np.array([])
    for r in range(len(halite_map)):
        for c in range(len(halite_map[0])):
            target_position = np.array((r,c)).astype(int)
            d = calculate_manhattan_distance(origin_position,target_position)
            d_array = np.append(d_array, d)
    d_array = np.reshape(d_array,(len(halite_map),len(halite_map[0])))

    # Quarter 4
    target_corner = np.array([len(halite_map)-1, len(halite_map[0])-1])
    mcps, mcp_quarter4, mcp_step = min_cost_path_to_a_cell(halite_map, target_corner)

    # Quarter 1
    target_corner = np.array([0, 0])
    mcps, mcp_quarter1, mcp_step = min_cost_path_to_a_cell(halite_map, target_corner)

    # Quarter 2
    target_corner = np.array([0, len(halite_map[0])-1])
    mcps, mcp_quarter2, mcp_step = min_cost_path_to_a_cell(halite_map, target_corner)

    # Quarter 3
    target_corner = np.array([len(halite_map)-1, 0])
    mcps, mcp_quarter3, mcp_step = min_cost_path_to_a_cell(halite_map, target_corner)

    # Combine the 4 quarters together
    full_mcp = np.zeros([len(halite_map), len(halite_map[0])])
    full_mcp[origin_position[0]:,origin_position[1]:] = mcp_quarter4
    full_mcp[0:origin_position[0]+1,0:origin_position[1]+1] = np.flip(np.flip(np.asarray(mcp_quarter1), axis=0), axis=1)
    full_mcp[0:origin_position[0]+1,origin_position[1]:] = np.flip(np.asarray(mcp_quarter2), axis=0)
    full_mcp[origin_position[0]:,0:origin_position[1]+1] = np.flip(np.asarray(mcp_quarter3), axis=1)

    # The Cost is effective only in the next move, hence it is post move and not prior move
    full_mcp = full_mcp - halite_map

    logging.info("create_roi_map: full_mcp =\n{}".format(np.array2string(full_mcp)))

    roi_array = np.array([])
    for r in range(len(halite_map)):
        for c in range(len(halite_map[0])):
            roi = halite_map[r,c] - (halite_map[r,c] * 0.75**(total-d_array[r,c])) - full_mcp[r,c] * 0.10 
            roi_array = np.append( roi_array, roi )
    roi_array = np.reshape(roi_array, (len(halite_map), len(halite_map[0])))
    roi_array = np.around(roi_array, decimals=0)
    logging.info("create_roi_map: roi_array = \n{}".format(np.array2string(roi_array)))

    max_roi_position = np.unravel_index(np.argmax(roi_array, axis=None), roi_array.shape)
    logging.info("create_roi_map: the max roi is {} at {}".format(roi_array[max_roi_position], max_roi_position))

    quarter = quarter_relative_to_origin(origin_position, max_roi_position)
    delta_row = abs(max_roi_position[0]-origin_position[0])
    delta_col = abs(max_roi_position[1]-origin_position[1])
    logging.info("create_roi_map: origin_position {}, max_roi_position {}, quarter {}".format(origin_position, max_roi_position, quarter))
    logging.info("create_roi_map: delta_row {}, and delta_col {}".format(delta_row, delta_col))
    if( quarter == 4 ):
        mcp_step = min_cost_path_step(mcp_quarter4[:delta_row+1, :delta_col+1]) * [-1, -1]
        mcp = mcp_quarter4
    if( quarter == 1 ):
        mcp_step = min_cost_path_step(mcp_quarter1[:delta_row+1, :delta_col+1]) * [1, 1]
        mcp = mcp_quarter1
    if( quarter == 2 ):
        mcp_step = min_cost_path_step(mcp_quarter2[:delta_row+1, :delta_col+1]) * [1, -1]
        mcp = mcp_quarter2
    if( quarter == 3 ):
        mcp_step = min_cost_path_step(mcp_quarter3[:delta_row+1, :delta_col+1]) * [-1, 1]
        mcp = mcp_quarter3

    logging.info("create_roi_map: mcp_step \n{}".format(mcp_step))

    # Take note, the mcp_step is last one to execute first to allow stack like operation for First In Last One (FILO)
    return mcps, mcp, mcp_step.tolist()


def create_lowest_cost_return(origin_position, target_position):
    origin_position = np.array([origin_position.y, origin_position.x])
    #target_position = np.array([me.shipyard.position.y, me.shipyard.position.x])
    target_position = np.array([target_position.y, target_position.x])
    logging.info("create_lowest_cost_return: origin_position = {}, target_position.position = {}".format(origin_position, target_position))
    quarter = quarter_relative_to_origin(origin_position, target_position)
    logging.info("create_lowest_cost_return: quarter = {}".format(quarter))

    grand_map = np.array([])
    for r in range( game.game_map.height ):
        for c in range( game.game_map.width ):
            h = game_map[Position(c, r)].halite_amount # Take note of X-Y convention in game_map, instead of Row-Col convention
            grand_map = np.append(grand_map, h)
    grand_map = np.reshape(grand_map, (game.game_map.height ,game.game_map.width))

    if( quarter == 4 ):
        sub_cost_map = grand_map[origin_position[0]:target_position[0]+1, origin_position[1]:target_position[1]+1]
        logging.info("create_lowest_cost_return: sub_cost_map = \n{}".format(np.array2string(sub_cost_map)))
        mcps, mcp = min_cost_path_whole(sub_cost_map)
        mcp_step = min_cost_path_step(mcp) * [-1, -1]

    if( quarter == 1 ):
        sub_cost_map = np.flip(np.flip(grand_map[target_position[0]:origin_position[0]+1, target_position[1]:origin_position[1]+1], axis=0), axis=1)
        logging.info("create_lowest_cost_return: sub_cost_map = \n{}".format(np.array2string(sub_cost_map)))
        mcps, mcp = min_cost_path_whole(sub_cost_map)
        mcp_step = min_cost_path_step(mcp) * [1, 1]

    if( quarter == 2 ):
        sub_cost_map = np.flip(grand_map[target_position[0]:origin_position[0]+1, origin_position[1]:target_position[1]+1], axis=0)
        logging.info("create_lowest_cost_return: sub_cost_map = \n{}".format(np.array2string(sub_cost_map)))
        mcps, mcp = min_cost_path_whole(sub_cost_map)
        mcp_step = min_cost_path_step(mcp) * [1, -1]

    if( quarter == 3 ):
        pass
        sub_cost_map = np.flip(grand_map[origin_position[0]:target_position[0]+1, target_position[1]:origin_position[1]+1], axis=1)
        logging.info("create_lowest_cost_return: sub_cost_map = \n{}".format(np.array2string(sub_cost_map)))
        mcps, mcp = min_cost_path_whole(sub_cost_map)
        mcp_step = min_cost_path_step(mcp) * [-1, 1]

    logging.info("create_lowest_cost_return: mcp_step = \n{}".format(np.array2string(mcp_step)))
    return mcp_step.tolist()    

def random_move(ship):
    move_choice = random.sample([ Direction.North, Direction.South, Direction.East, Direction.West ], 4)

    for x in range(len(move_choice)):
        newPosition = ship.position.directional_offset(move_choice[x])

        if( unsafe_map[ newPosition.y, newPosition.x ] == 0 ):
            return move_choice[x]

    # Exhause all move directions, only left standing still as the only option
    return(Direction.Still)


def mark_enemy_ship_on_unsafe_map():
    for player_id in game.players:
        if(player_id != game.my_id ) :
            for ship in game.players[player_id].get_ships():
                logging.info("mark_enemy_ship_on_unsafe_map: player_id={}, ship.id={}, ship.position={}".format(player_id, ship.id, ship.position))
                unsafe_map[ ship.position.y, ship.position.x ] = 1
                for cardinal in ship.position.get_surrounding_cardinals():
                    logging.info("mark_enemy_ship_on_unsafe_map: player_id={}, ship.id={}, ship.position={}, cardinal={}".format(player_id, ship.id, ship.position, cardinal))
                    unsafe_map[ cardinal.y, cardinal.x ] = 1

    # It means, the enemy ship is on our shipyard, clear the unsafe_map to get rid of enemy ship
    if( unsafe_map[me.shipyard.position.y, me.shipyard.position.x] == 1 ):
        unsafe_map[me.shipyard.position.y, me.shipyard.position.x] = 0
            

class RevolzShip():
    def __init__(self, ship):
        self.ship = ship
        self.state = "SEARCHING"
        self.radius = 3
        self.enough = 0.8
        self.halite_map = np.array([])
        self.halite_total = 0
        self.halite_average = 0
        self.halite_max = 0
        self.halite_max_position = 0
        self.mcps = 0
        self.mcp = 0
        self.mcp_step = 0


    def log_state(self):
        logging.info("----- Ship {} at {} -----".format(self.ship.id, self.ship.position))
        logging.info("Halite Map = \n{}".format(np.array2string(self.halite_map)))
        logging.info("Total Halite = {}".format(self.halite_total))
        logging.info("Average Halite = {}".format(self.halite_average))
        logging.info("Max Halite Amount = {}".format(self.halite_max))
        logging.info("Max Halite Position = {}".format(self.halite_max_position))
        logging.info("Min Cost Path Sum = {}".format(self.mcps))
        logging.info("Min Cost Path = \n{}".format(np.array2string(self.mcp)))
        logging.info("Min Cost Path Step (last one to execute first) = \n{}".format(self.mcp_step))
        
    def create_map_awareness2(self, radius):
        self.radius = radius

        # Sense the map
        total = 0
        self.halite_map = np.array([])
        for r2 in range( -radius, radius+1 ):
            for r1 in range( -radius, radius+1 ):
                h = game_map[Position(self.ship.position.x+r1, self.ship.position.y+r2)].halite_amount
                self.halite_map = np.append(self.halite_map, h)
                total += h

        # Calculate the result
        self.halite_map = np.reshape(self.halite_map,(radius*2+1,radius*2+1))        
        self.halite_total = total
        self.halite_average = total / ((radius*2+1)*(radius*2+1))
        self.halite_max_position = np.array(np.unravel_index(self.halite_map.argmax(), self.halite_map.shape))
        self.halite_max = game_map[
            Position(
                self.ship.position.x+self.halite_max_position[1]-self.radius, 
                self.ship.position.y+self.halite_max_position[0]-self.radius)
            ].halite_amount


    def create_map_awareness(self, radius):
        self.radius = radius

        # Sense the map
        total = 0
        self.halite_map = np.array([])
        for r2 in range( -radius, radius+1 ):
            for r1 in range( -radius, radius+1 ):
                h = game_map[Position(self.ship.position.x+r1, self.ship.position.y+r2)].halite_amount
                self.halite_map = np.append(self.halite_map, h)
                total += h

        # Calculate the result
        self.halite_map = np.reshape(self.halite_map,(radius*2+1,radius*2+1))        
        self.halite_total = total
        self.halite_average = total / ((radius*2+1)*(radius*2+1))
        self.halite_max_position = np.array(np.unravel_index(self.halite_map.argmax(), self.halite_map.shape))
        self.halite_max = game_map[
            Position(
                self.ship.position.x+self.halite_max_position[1]-self.radius, 
                self.ship.position.y+self.halite_max_position[0]-self.radius)
            ].halite_amount
        #self.mcps, self.mcp, self.mcp_step = min_cost_path_to_a_cell(self.halite_map, self.halite_max_position)
        self.mcps, self.mcp, self.mcp_step = create_roi_map(self.halite_map)

        # Output the state
        # self.log_state()

    def next_command(self):
        logging.info("----- Ship {} at {} -----".format(self.ship.id, self.ship.position))

        loop_time = 10
        while( loop_time > 0 ):
            loop_time = loop_time - 1

            # End Game
            if( self.state != "RETURNING" and game_map.calculate_distance(ship.position, me.shipyard.position) >= (constants.MAX_TURNS - game.turn_number) - 5 ):
                logging.info("next_command: state == ENDING for Ship {}".format(self.ship.id))
                self.mcp_step = create_lowest_cost_return(self.ship.position, get_nearest_return_position(self.ship.position))
                self.state = "RETURNING"

            if( self.state == "SEARCHING" ):
                logging.info("next_command: state == SEARCHING for Ship {}".format(self.ship.id))
                self.create_map_awareness(self.radius)
                logging.info("next_command: state == SEARCHING: DONE for Ship {}".format(self.ship.id))
                self.state = "NAVIGATING"
            
            if( self.state == "NAVIGATING" ):
                logging.info("next_command: state == NAVIGATING for Ship {}".format(self.ship.id))

                if( len(self.mcp_step) > 0 ):
                    if( self.ship.halite_amount < game_map[self.ship.position].halite_amount * 0.10 ):
                        logging.info("next_command: state == NAVIGATING for Ship {}, but not enough halite to move".format(self.ship.id))
                        next_command = self.ship.stay_still()
                        
                        # Mark the intended position as unsafe
                        unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1

                        return next_command
                    
                    else:
                        next_move = self.mcp_step.pop()
                        
                        logging.info("collision avoidance: ship.position.y {}, next_move[0] {}, ship.position.x {}, next_move[1] {}".format(self.ship.position.y, next_move[0], self.ship.position.x, next_move[1]))
                        logging.info("collision avoidance: unsafe_map at the new position {}".format(unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ])) # comment out for debugging, not sure what has gone wrong
                        # If the intended position is safe
                        if( unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] == 0 ):
                            next_command = self.ship.move(
                                Direction.convert(
                                    ( next_move[1], next_move[0] ) # NumPy is Row-Col based & Halite Map is X-Y based. It requires conversion between the two.
                                )
                            )
                            
                            # Mark the intended position as unsafe
                            unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] = 1
                            logging.info("next_command: updated unsafe_map for Ship {}".format(self.ship.id))
                            logging.info("next_command: unsafe_map[{},{}] = {}".format(
                                (self.ship.position.y + next_move[0]) % game.game_map.height, 
                                (self.ship.position.x + next_move[1]) % game.game_map.width, 
                                unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ]))
                            return next_command
                        
                        # If the intended position is unsafe
                        else:
                            # Push back the next move
                            self.mcp_step.append(next_move)

                            logging.info("next_command: state == NAVIGATING for Ship {}, but intended position is unsafe".format(self.ship.id))
                            if( unsafe_map[ self.ship.position.y, self.ship.position.x ] == 0 ):
                                next_command = self.ship.stay_still()
                                
                                # Mark the intended position as unsafe
                                unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1
                            
                            else:
                                next_move = random_move(self.ship)
                                logging.info("MINING: unsafe to continue navigating - next_move = {}".format(next_move))
                                next_command = self.ship.move( next_move )

                                 # Mark the intended position as unsafe
                                unsafe_map[ (self.ship.position.y + next_move[1]) % game.game_map.height, (self.ship.position.x + next_move[0]) % game.game_map.width ] = 1

                                self.state = "SEARCHING"

                            return next_command

                else:
                    self.state = "MINING"

            if( self.state == "MINING" ):
                logging.info("next_command: state == MINING for Ship {}".format(self.ship.id))

                if( self.ship.is_full or self.ship.halite_amount > constants.MAX_HALITE * self.enough ):
                    self.mcp_step = create_lowest_cost_return(self.ship.position, get_nearest_return_position(self.ship.position))
                    self.state = "RETURNING"
                
                elif( game_map[self.ship.position].halite_amount < constants.MAX_HALITE / 20 ):
                    self.state = "SEARCHING"

                    logging.info("next_command: state == Changing to SEARCHING for Ship {}".format(self.ship.id))  
                    logging.info("next_command: unsafe_map[{},{}] = {}".format(
                                self.ship.position.y, 
                                self.ship.position.x, 
                                unsafe_map[ self.ship.position.y, self.ship.position.x]))

                    if( unsafe_map[ self.ship.position.y, self.ship.position.x ] == 0 ):
                        next_command = self.ship.stay_still()
                    
                        # Mark the intended position as unsafe
                        unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1
                    
                    else:
                        next_move = random_move(self.ship)
                        logging.info("MINING: unsafe to continue mining - next_move = {}".format(next_move))
                        next_command = self.ship.move( next_move )

                         # Mark the intended position as unsafe
                        unsafe_map[ (self.ship.position.y + next_move[1]) % game.game_map.height, (self.ship.position.x + next_move[0]) % game.game_map.width ] = 1
                        #unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] = 1
                    
                    return next_command

                else:
                    logging.info("next_command: state == MINING for Ship {}".format(self.ship.id))
                    logging.info("next_command: unsafe_map[{},{}] = {}".format(
                                self.ship.position.y, 
                                self.ship.position.x, 
                                unsafe_map[ self.ship.position.y, self.ship.position.x]))

                    if( unsafe_map[ self.ship.position.y, self.ship.position.x ] == 0 ):
                        next_command = self.ship.stay_still()
                    
                        # Mark the intended position as unsafe
                        unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1
                    
                    else:
                        next_move = random_move(self.ship)
                        logging.info("MINING: unsafe to continue mining - next_move = {}".format(next_move))
                        next_command = self.ship.move( next_move )

                         # Mark the intended position as unsafe
                        unsafe_map[ (self.ship.position.y + next_move[1]) % game.game_map.height, (self.ship.position.x + next_move[0]) % game.game_map.width ] = 1
                        # unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] = 1

                    return next_command


            if( self.state == "RETURNING" ):
                if( self.ship.position == get_nearest_return_position(self.ship.position) ):
                    self.state = "SEARCHING"

                    # Not needed, keep to monitor further
                    # Mark the intended position as unsafe
                    # unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1
                
                else:
                    if( len(self.mcp_step) > 0 ):
                        if( self.ship.halite_amount < game_map[ship.position].halite_amount * 0.10 ):
                            logging.info("next_command: state == NAVIGATING for Ship {}, but not enough halite to move".format(self.ship.id))
                            next_command = self.ship.stay_still()

                            # Mark the intended position as unsafe
                            unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1

                            return next_command

                        else:
                            # next_command = self.ship.move( game_map.naive_navigate(self.ship, me.shipyard.position) )
                            next_move = self.mcp_step.pop()
                            
                            # If the intended position is safe
                            if( unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] == 0 ):
                                next_command = self.ship.move(
                                    Direction.convert(
                                        ( next_move[1], next_move[0] ) # NumPy is Row-Col based & Halite Map is X-Y based. It requires conversion between the two.
                                    )
                                )
                                
                                # Mark the intended position as unsafe
                                unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] = 1
                                logging.info("next_command: state == RETURNING for Ship {}".format(self.ship.id))
                                return next_command
                            
                            # If the intended position is unsafe
                            else:
                                # Push back the next move
                                self.mcp_step.append(next_move)

                                logging.info("next_command: state == RETURNING for Ship {}, but intended position is unsafe".format(self.ship.id))
                                if( unsafe_map[ self.ship.position.y, self.ship.position.x ] == 0 ):
                                    next_command = self.ship.stay_still()
                                    
                                    # Mark the intended position as unsafe
                                    unsafe_map[ self.ship.position.y, self.ship.position.x ] = 1
                                
                                else:
                                    next_move = random_move(self.ship)
                                    logging.info("MINING: unsafe to continue returning - next_move = {}".format(next_move))
                                    next_command = self.ship.move( next_move )

                                    # Mark the intended position as unsafe
                                    unsafe_map[ (self.ship.position.y + next_move[1]) % game.game_map.height, (self.ship.position.x + next_move[0]) % game.game_map.width ] = 1
                                    #unsafe_map[ (self.ship.position.y + next_move[0]) % game.game_map.height, (self.ship.position.x + next_move[1]) % game.game_map.width ] = 1

                                    self.state = "MINING"

                                return next_command


# Returning the maximum ship quantity among enem(ies)
def get_max_enemy_ship():
    max_enemy_ship = 0

    for player_id in game.players :
        if(player_id != game.my_id ) :
            if(len(game.players[player_id].get_ships()) > max_enemy_ship ) :
                max_enemy_ship = len(game.players[player_id].get_ships())

    return max_enemy_ship


# Returning the ship count of the enemy with the most halite
def get_richest_enemy_ship_count():
    richest_enemy_halite = 0
    richest_enemy_ship_count = 0

    for player_id in game.players:
        if(player_id != game.my_id ) :
            if( game.players[player_id].halite_amount > richest_enemy_halite ) :
                richest_enemy_halite = game.players[player_id].halite_amount
                richest_enemy_ship_count = len(game.players[player_id].get_ships())

    return richest_enemy_ship_count


def get_nearest_return_position(origin_position):
    distance_to_shipyard = game_map.calculate_distance( origin_position, me.shipyard.position )
    nearest_return_position = 0

    if( len( me.get_dropoffs() ) > 0 ):
        distance_to_dropoff = game_map.calculate_distance( origin_position, me.get_dropoffs()[0].position )
    
        logging.info("get_nearest_return_position: shipyard distance {}, vs dropoff distance {}".format(distance_to_shipyard, distance_to_dropoff))
        if( distance_to_shipyard < distance_to_dropoff ):
            nearest_return_position = me.shipyard.position
        else:
            nearest_return_position = me.get_dropoffs()[0].position
    
    else:
        nearest_return_position = me.shipyard.position

    logging.info("get_nearest_return_position: {}".format(nearest_return_position))
    return nearest_return_position


revolz_fleet = {}

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("RevolzBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

dropoff_allowance = 1

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    # A unsafe map
    # Two types, intended location of our own ships, potential location of enemy ships
    unsafe_map = np.zeros((game.game_map.height, game.game_map.width))
    mark_enemy_ship_on_unsafe_map()

    # If game progress is before 85%, and you are less than richest enemy ships count + 5 and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if( game.turn_number < constants.MAX_TURNS * 0.85 and len(me.get_ships()) <= get_richest_enemy_ship_count() + 5 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied ):
        command_queue.append(me.shipyard.spawn())
        unsafe_map[me.shipyard.position.y, me.shipyard.position.x] = 1
        logging.info("ship spawn: game.turn_number={}, constants.MAX_TURNS={},richest_enemy_ship_count={}".format(game.turn_number, constants.MAX_TURNS, get_richest_enemy_ship_count()))

    ship_list = me.get_ships()

    # If it meets the dropoff creation criteria
    if( dropoff_allowance > 0 and game.turn_number > constants.MAX_TURNS * 0.45 and me.halite_amount > 5000 ):
        dropoff_allowance = dropoff_allowance - 1
        
        most_halite = 0
        ship_with_most_halite = 0

        for ship in ship_list:
            if ship.id in revolz_fleet:
                if( revolz_fleet[ship.id].halite_total > most_halite and ship.position != me.shipyard.position ):
                    revolz_fleet[ship.id].create_map_awareness2(3)
                    most_halite = revolz_fleet[ship.id].halite_total
                    ship_with_most_halite = ship
                    logging.info("Current best dropoff with Ship {} with Total Halite {} at {}".format(ship_with_most_halite.id, revolz_fleet[ship_with_most_halite.id].halite_total, ship_with_most_halite.position))

        logging.info("Finalize dropoff with Ship {} with Total Halite {} at {}".format(ship_with_most_halite.id, revolz_fleet[ship_with_most_halite.id].halite_total, ship_with_most_halite.position))
        ship_list.remove( ship_with_most_halite )
        command_queue.append( ship_with_most_halite.make_dropoff() )    

    # Classify non-movable & movable ships
    non_movable_ships = []
    movable_ships = []
    for ship in ship_list:
        if( ship.halite_amount < game_map[ship.position].halite_amount * 0.10 ):
            non_movable_ships.append(ship)
        
        else:
            movable_ships.append(ship)

    for ship in non_movable_ships:
        
        # Using revolz_fleet dictionary to maintain the ship fleet
        if ship.id not in revolz_fleet:
            revolz_fleet[ship.id] = RevolzShip(ship)
        
        # Append the next command for the ship
        next_command = revolz_fleet[ship.id].next_command()
        logging.info("next_command = {}".format(next_command))
        if( next_command != None ):
            command_queue.append( next_command )

    for ship in movable_ships:
        
        # Using revolz_fleet dictionary to maintain the ship fleet
        if ship.id not in revolz_fleet:
            revolz_fleet[ship.id] = RevolzShip(ship)
        
        # Append the next command for the ship
        next_command = revolz_fleet[ship.id].next_command()
        logging.info("next_command = {}".format(next_command))
        if( next_command != None ):
            command_queue.append( next_command )
          
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
