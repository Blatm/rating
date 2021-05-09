#!/usr/bin/python -tt

import sys
#from sage.all import *
import cProfile
import numpy as np



############
### Main ###
############

def main():
    pass


###############
### Reading ###
###############

'''
https://database.lichess.org/
'''

def lichess_pgn_to_data(filename, to_record=['WhiteElo', 'BlackElo', 'Result']):
    '''
    data = rating.lichess_pgn_to_data('./data/lichess/lichess_db_standard_rated_2013-01.pgn')
    '''
    CHARACTERS_TO_REMOVE = ['"', '\n', '[', ']']
    
    data = []
    f = open(filename, 'r')
    datum = None
    for line in f:
        # start new datum
        if (len(line) > 6) and (line[:6] == '[Event'):
            if datum is not None:
                data.append(datum)
            datum = {}
        
        if (len(line) > 0) and (line[0] == '['):
            split_line = line.split(' ')
            key = split_line[0].strip('[')
            if key in to_record:
                val = ''
                for piece in split_line[1:]:
                    for badchar in CHARACTERS_TO_REMOVE:
                        piece = piece.replace(badchar, '')
                    val += piece
                    val += ' '
                val = val[:-1]
                datum[key] = val
    
    if datum is not None:
        data.append(datum)
    f.close()
    
    return data


def get_cleaned_elos_and_results(data):
    '''
    data from lichess_pgn_to_data with to_record containing 'WhiteElo', 'BlackElo', 'Result'
    Returns a list of tuples (white_elo, black_elo, result), where result is '1-0', '0-1', or '1/2-1/2'
    '''
    tups = []
    for datum in data:
        try:
            white_elo = int(datum['WhiteElo'])
            black_elo = int(datum['BlackElo'])
            result = datum['Result']
            tups.append((white_elo, black_elo, result))
        except:
            continue
    return tups
    

###############
### Binning ###
###############

def make_diff_data(data, min_elo=None, max_elo=None):
    '''
    data is expected to be a list of (white_elo, black_elo, result)
    Returns a list of (white_elo - black_elo, result) with elos between min_elo and max_elo
    '''
    diff_data = []
    for (white_elo, black_elo, result) in data:
        if (min_elo is None) or ((white_elo > min_elo) and (black_elo > min_elo)):
            if (max_elo is None) or ((white_elo < max_elo) and (black_elo < max_elo)):
                diff = white_elo - black_elo
                diff_data.append((diff, result))
    return diff_data


def make_winrate_data(diff_data, bin_size=5):
    '''
    Returns a dict of {binned elo diff: (winrate, error)}, where
      binned elo diff is white_elo - black_elo, rounded to the nearest bin_size
    '''
    binned_data = get_binned_data(diff_data, bin_size=bin_size)
    winrate_data = {}
    for bin_val in binned_data:
        game_results = binned_data[bin_val]
        num_games = len(game_results)
        winrate = np.sum(game_results)/float(num_games)
        error = 1.0/np.sqrt(float(num_games))
        winrate_data[bin_val] = (winrate, error)
    return winrate_data


def get_binned_data(diff_data, bin_size=5):
    bin_size_int = int(bin_size)
    bin_size_float = float(bin_size)
    binned_data = {}
    for (diff, result) in diff_data:
        bin_val = round(diff/bin_size_float) * bin_size_int
        if bin_val not in binned_data:
            binned_data[bin_val] = []
        if result == '1-0':
            binned_data[bin_val].append(1)
        elif result == '0-1':
            binned_data[bin_val].append(0)
        elif result == '1/2-1/2':
            binned_data[bin_val].append(0.5)
        else:
            raise ValueError('In get_binned_data, got result '+str(result))
    return binned_data


def file_to_winrates(filename, bin_size=5, min_elo=None, max_elo=None):
    '''
    Returns a dict of {binned elo diff: (winrate, error)}, where
      binned elo diff is white_elo - black_elo, rounded to the nearest bin_size
    Data in filename is filtered so that elos are between min_elo and max_elo
    '''
    data = lichess_pgn_to_data(filename)
    cleaned_data = get_cleaned_elos_and_results(data)
    diff_data = make_diff_data(cleaned_data, min_elo=min_elo, max_elo=max_elo)
    winrate_data = make_winrate_data(diff_data, bin_size=bin_size)
    return winrate_data
    

#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
