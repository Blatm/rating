#!/usr/bin/python -tt

import sys
#from sage.all import *
import cProfile
import numpy as np
import random



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

def default_filename():
    '''
    For convenience
    '''
    fn = './data/lichess/lichess_db_standard_rated_2013-01.pgn'
    return fn


def all_lichess_keys():
    all_keys = ['Event', 'Site', 'White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo', 'WhiteRatingDiff', 'BlackRatingDiff', 'ECO', 'Opening', 'TimeControl', 'Termination']
    return all_keys


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


def result_str_to_float(result_str):
    if result_str == '1-0':
        result_float = 1.0
    elif result_str == '0-1':
        result_float = 0.0
    elif result_str == '1/2-1/2':
        result_float = 0.5
    else:
        raise ValueError('In result_str_to_float, got result '+str(result_str))
    return result_float


def make_winrate_data(diff_data, num_samples_per_bin=400, std_only=True):
    '''
    Returns a dict of {binned elo diff: (winrate, error)}, where
      binned elo diff is white_elo - black_elo, averaged across the bin,
      and error is the std of the diffs making the bin if std_only, and the list of diffs in the bin otherwise
    '''
    diff_data = discard_extras(diff_data, num_samples_per_bin) # len(diff_data) % num_samples_per_bin == 0 after this
    diff_data.sort(key = lambda tup: tup[0]) # sort by white_elo - black elo in ascending order
    winrate_data = {}
    for data_chunk_counter in range(len(diff_data)/num_samples_per_bin):
        diff_samples = diff_data[num_samples_per_bin*data_chunk_counter : num_samples_per_bin*(data_chunk_counter+1)]
        # diff samples is a list of tups (elo_diff, result), where result is '1-0', '0-1', or '1/2-1/2'
        chunk_results = []
        for (diff, result_str) in diff_samples:
            result = result_str_to_float(result_str)
            chunk_results.append(result)
        binned_elo_diff = np.sum([datum[0] for datum in diff_samples])/float(len(diff_samples))
        chunk_winrate = np.sum(chunk_results)/float(len(chunk_results))
        if std_only:
            error = np.std([tup[0] for tup in diff_samples])
        else:
            error = [tup[0] for tup in diff_samples]
        winrate_data[binned_elo_diff] = (chunk_winrate, error) # in principle I should check that this key is unique...
    return winrate_data


def discard_extras(data, num_samples_per_bin):
    num_to_discard = len(data) % num_samples_per_bin
    index_list = list(range(len(data)))
    random.shuffle(index_list)
    indices_to_discard = index_list[:num_to_discard]
    indices_to_discard.sort(reverse=True)
    new_data = list(data)
    for index in indices_to_discard:
        del new_data[index]
    return new_data


def make_winrate_data_fixed_bins(diff_data, bin_size=5):
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
        result_float = result_str_to_float(result)
        binned_data[bin_val].append(result_float)
    return binned_data


def file_to_winrates_fixed_bins(filename, bin_size=5, min_elo=None, max_elo=None):
    '''
    Returns a dict of {binned elo diff: (winrate, error)}, where
      binned elo diff is white_elo - black_elo, rounded to the nearest bin_size
    Data in filename is filtered so that elos are between min_elo and max_elo
    '''
    data = lichess_pgn_to_data(filename)
    cleaned_data = get_cleaned_elos_and_results(data)
    diff_data = make_diff_data(cleaned_data, min_elo=min_elo, max_elo=max_elo)
    winrate_data = make_winrate_data_fixed_bins(diff_data, bin_size=bin_size)
    return winrate_data


def file_to_winrates(filename, num_samples_per_bin=400, std_only_errors=True, min_elo=None, max_elo=None):
    '''
    Returns a dict of {binned elo diff: (winrate, error)}, where
      binned elo diff is white_elo - black_elo, rounded to the nearest bin_size
    Data in filename is filtered so that elos are between min_elo and max_elo
    '''
    data = lichess_pgn_to_data(filename)
    cleaned_data = get_cleaned_elos_and_results(data)
    diff_data = make_diff_data(cleaned_data, min_elo=min_elo, max_elo=max_elo)
    winrate_data = make_winrate_data(diff_data, num_samples_per_bin=num_samples_per_bin, std_only=std_only_errors)
    return winrate_data


######################
### Winrate tables ###
######################

def make_winrate_table(filename):
    '''
    Returns a dict with keys (player1, player2) and values {'numgames': numgames between p1 and p2, 'player1_winrate': average score of p1 vs p2}
    The keys are always sorted alphabetically. This doesn't keep track of who was white and who was black.
    '''
    to_record = ['White', 'Black', 'Result']
    data = lichess_pgn_to_data(filename, to_record=to_record)
    winrate_table = {}
    for game in data:
        white = game['White']
        black = game['Black']
        result = game['Result']
        players = [white, black]
        reverse = (sorted(players)[0] != players[0])
        if reverse:
            if result == '1-0':
                result = '0-1'
            elif result == '0-1':
                result = '1-0'
        if result == '1-0':
            player1_score = 1.0
        elif result == '0-1':
            player1_score = 0.0
        elif result == '1/2-1/2':
            player1_score = 0.5
        else:
            raise ValueError('In make_winrate_table, got result '+str(result))
        players = tuple(sorted(players))
        if players not in winrate_table:
            winrate_table[players] = {'numgames':1, 'player1_winrate':player1_score}
        else:
            old_numgames = winrate_table[players]['numgames']
            old_player1_winrate = winrate_table[players]['player1_winrate']
            new_player1_winrate = (old_player1_winrate * old_numgames + player1_score)/(old_numgames + 1)
            winrate_table[players]['numgames'] = old_numgames + 1
            winrate_table[players]['player1_winrate'] = new_player1_winrate
    return winrate_table


def make_opponent_dict(winrate_table):
    opponent_dict = {}
    for player_tup in winrate_table:
        for player in player_tup:
            if player not in opponent_dict:
                opponent_dict[player] = set()
        opponent_dict[player_tup[0]].add(player_tup[1])
        opponent_dict[player_tup[1]].add(player_tup[0])
    return opponent_dict


def get_nonempty_triples(opponent_dict, winrate_table):
    triples = {}
    for p in opponent_dict:
        for q in opponent_dict[p]:
            for r in opponent_dict[q]:
                if r in opponent_dict[p]:
                    trip = [p,q,r]
                    trip = tuple(sorted(trip))
                    (ps,qs,rs) = trip
                    p_vs_q = winrate_table[(ps,qs)]
                    p_vs_r = winrate_table[(ps,rs)]
                    q_vs_r = winrate_table[(qs,rs)]
                    triples[trip] = (p_vs_q, p_vs_r, q_vs_r)
    return triples


def get_useful_triples(triples):
    '''
    Returns a dict (p,q,r): (p_vs_q, p_vs_r, q_vs_r)
    The keys (p,q,r) are the triples for which the winrates aren't all 0.0 or 1.0
    '''
    useful_triples = {}
    for trip in triples:
        p_vs_q = triples[trip][0]
        p_vs_r = triples[trip][1]
        q_vs_r = triples[trip][2]
        if p_vs_q*(1-p_vs_q) + p_vs_r*(1-p_vs_r) + q_vs_r*(1-q_vs_r) > 0:
            useful_triples[trip] = (p_vs_q, p_vs_r, q_vs_r)
    return useful_triples


def file_to_nonempty_triples(filename):
    '''
    Returns a dict where
      the keys are triples of players (p,q,r) in alphabetical order
      the values are tuples (p_vs_q, p_vs_r, q_vs_r),
        where p1_vs_p2 are dicts {'numgames': numgames between p1 and p2, 'player1_winrate': average score of p1 vs p2}
      the only keys present are triples (p,q,r) for which all numgames are nonzero
    '''
    winrate_table = make_winrate_table(filename)
    opponent_dict = make_opponent_dict(winrate_table)
    triples = get_nonempty_triples(opponent_dict, winrate_table)
    return triples


#########################
### Bayesian winrates ###
#########################


class WinrateDistribution(dict):
    '''
    This models a distribution of a true winrate.
    It's a dict, and the keys are winrates, and the values are probabilities
    '''
    def __init__(self,*args,**kwargs):
        if 'filename' in kwargs:
            self.build_from_file(**kwargs)
        else:
            self = super(WinrateDistribution, self).__init__(*args, **kwargs)
    
    
    def __repr__(self):
        to_ret = ''
        for key in sorted(self):
            to_ret += str(key)+': '+str(self[key])+'\n'
        if len(to_ret) > 0:
            to_ret = to_ret[:-1]
        return to_ret
    

    def build_from_file(self, filename, num_samples_per_bin=400, min_elo=None, max_elo=None):
        '''
        This generates a distribution based on the data in filename.
        file_to_winrates has a constant number of games per bin,
        so it's essentially answering the question of "given that two players are playing, what's the distribution of the true winrate?"
        '''
        winrate_dict = file_to_winrates(filename, num_samples_per_bin=num_samples_per_bin, min_elo=min_elo, max_elo=max_elo)
        for elo_diff in winrate_dict:
            (winrate, error) = winrate_dict[elo_diff]
            self[winrate] = 1.0
        self.normalize()
    
    
    def normalize(self):
        normalization_constant = self.kth_moment(0)
        for key in self:
            self[key] = self[key]/normalization_constant

    
    def update(self, result):
        # result == 1 for win, 0 for loss
        if type(result) == type(''):
            result = result_str_to_float(result)
        for key in self:
            self[key] *= key**result * (1-key)**(1-result)
        self.normalize()


    def kth_moment(self, k):
        moment_val = 0.0
        for key in self:
            moment_val += key**k * self[key]
        return moment_val


    def avg(self):
        return self.kth_moment(1)


    def std(self):
        return self.kth_moment(2)



#This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    if '-profile' in sys.argv:
        cProfile.run('main()')
    else:
        main()
