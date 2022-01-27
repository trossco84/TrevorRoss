import pandas as pd
import numpy as np

def play_game():
    word = input("What is your word? ")
    for i in range(0,20):
        print("*")
    initial_blanks = " _ "*len(word)
    
    ready_check = "n"
    while ready_check not in ["yes","y","Yes","Y"]:
        ready_check = input("Is Player Ready? ")
    
    print("Let's Start!")
    print("You have 6 chances")
    print()
    print("Your word:")
    print(initial_blanks)

    strikes = 0
    guesses = []
    bad_guesses = []
    while strikes < 6:
        print(f"Strikes Left: {6 - strikes}")
        print()
        guess = input("Guess a Letter! ")
        print()
        while (len(guess)>1):
            print("Please input only one letter")
            print(f"Current Correct Guesses: {guesses}")
            print(f"Current Incorrect Guesses: {bad_guesses}")
            guess = input("Guess a Letter! ")
        
        while (guess in bad_guesses) or (guess in guesses):
            print("You've already guessed that!")
            print(f"Current Correct Guesses: {guesses}")
            print(f"Current Incorrect Guesses: {bad_guesses}")
            guess = input("Guess a Letter! ")
        
        if guess in word:
            print("Correct!")
            guesses.append(guess)
            print()
            
        else:
            print("STRIKE! That letter is not in the word")
            bad_guesses.append(guess)
            strikes = strikes+1
            print()

        word_split = list(word)
        for letter in word_split:
            if letter not in guesses:
                w_index = word_split.index(letter)
                word_split[w_index] = " _ "
        new_blanks = " "
        for l2 in word_split:
            new_blanks = new_blanks + l2

        if " _ " not in word_split:
            print("You've done it! You Win!!")
            print(new_blanks)
            exit()
        else:
            print("Your current word:")
            print(new_blanks)
    print("You LOSE!")
    print(f"the word was: {word}")
            
if __name__ == "__main__":
    play_game()