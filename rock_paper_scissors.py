import random
#Create a function
def RPS_game() -> None:
    """Asks for player and computer weapons. Then based on what each player picked, print
    out the message that corresponds to the result; you win, lose, tie, or pick a new weapon."""
    #Computer chooses randomly, player picks a weapon
    comp_choice = random.choice(["rock", "paper", "scissors"])
    player_choice = input("Choose rock, paper, or scissors: ")
    #If weapon illegal, asks you to pick a legal weapon
    if not (is_legal(player_choice)):
        print("You must select from rock, paper, or scissors")
    else:
        #If you beat the computer, you win.
        print("The computer chose", comp_choice)
        if beats(player_choice, comp_choice):
            print("You win!")
        #If not you lose    
        elif beats(comp_choice, player_choice):
            print("You lost. :(")
        #If weapons are the same it's a tie.    
        else:
            print("It's a tie.")

            
def is_legal(weapon:str)->bool:
    """Ensures the chosen weapon is "legal"."""
    if weapon == "rock" or "paper" or "scissors":
        return "true"
    return "false"


def beats(weapon1:str,weapon2:str) -> bool:
    """Defines what "beats" what. Win = true. Else false"""
    if weapon1 == "rock" and weapon2 == "rock":
        return (False)
    if weapon1 == "rock" and weapon2 == "paper":
        return (False)
    if weapon1 == "rock" and weapon2 == "scissors":
        return (True)
    if weapon1 == "paper" and weapon2 == "rock":
        return (True)
    if weapon1 == "paper" and weapon2 == "paper":
        return (False)
    if weapon1 == "paper" and weapon2 == "scissors":
        return (False)
    if weapon1 == "scissors" and weapon2 == "rock":
        return (False)
    if weapon1 == "scissors" and weapon2 == "paper":
        return (True)
    if weapon1 == "scissors" and weapon2 == "scissors":
        return (False)
    

    
