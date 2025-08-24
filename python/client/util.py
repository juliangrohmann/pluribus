from termcolor import colored
from colorama import just_fix_windows_console

def round_to_str(r:int) -> str: return ("Preflop", "Flop", "Turn", "River", "Showdown")[r]
def colorize_board(board:str) -> str: return ''.join(colored(board[i:i+2], {'s': "light_grey", 'h': "red", 'c': "green", 'd': "blue"}[board[i+1]]) for i in range(0, len(board), 2))