##//##//##//## Welcome to blackjack ##//##//##//##

#Importing necessary libraries
import numpy as np
import pandas as pd 
import requests 
import json 
import time

#URLs
# create_deck_url = "https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=6"
# draw_cards_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/draw/?count=2"
# shuffle_deck_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/shuffle/"

#Create a deck
create_deck_url = "https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=6"
response_data = requests.get(create_deck_url).json()
deck_id = response_data['deck_id']


#Draw Cards from Deck
def draw_card():
    draw_cards_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/draw/?count=1"
    card = requests.get(draw_cards_url).json()
    return card

#Shuffle Cards
shuffle_deck_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/shuffle/"

#Placing bets
def place_bets(players):
    print("Place Your Bets!")
    wagerlist=[]
    for player in players:
        wager = input(f"{player}'s Wager:")
        wagerlist.append(wager)
    return wagerlist

#Deal Cards
def deal_cards(cardsdf,gamenumber):
    cards_list = []
    for pxcy in cardsdf['Card1']:
        newcard = draw_card()
        newcardvalue = newcard['cards'][0]['value']
        if pxcy != "DealerCard2":
            print(f'{pxcy}: {newcardvalue}')
        else:
            print(f"{pxcy} Hidden")
        time.sleep(0.2)
        cards_list.append(newcardvalue)
    for pxcy in cardsdf['Card2']:
        newcard = draw_card()
        newcardvalue = newcard['cards'][0]['value']
        if pxcy != "DealerCard2":
            print(f'{pxcy}: {newcardvalue}')
        else:
            print(f"{pxcy} Hidden")
        time.sleep(0.2)
        cards_list.append(newcardvalue)
    return cards_list


def show_the_board(currentgame):
    print("Current Board")
    tempdf = pd.DataFrame()
    tempdf['CardSlots']=currentgame['CardSlots']
    tempdf['CurrentHand'] = currentgame.iloc[:,-1]
    dealer_showing = tempdf.iloc[-2,1]
    print(f'Dealer Showing: {dealer_showing}')

    num_player_cards = len(tempdf) - 2
    num_players = int(num_player_cards/2)
    for i in range(0,num_players):
        playerhand = []
        card1_loc = 2*i
        card2_loc = (2*i)+1
        card1 = tempdf['CurrentHand'][card1_loc]
        card2 = tempdf['CurrentHand'][card2_loc]
        playerhand.append(card1)
        playerhand.append(card2)
        print(f"Player{i+1}'s Hand:{playerhand}")

def show_the_round(current_round):

def play_a_round(players,cards):
    playersanddealer = players.copy()
    cards = cards_dealt.copy()
    playersanddealer.append("Dealer")
    starting_hand = cards.iloc[:,-1]
    rounddf = pd.DataFrame(data=playersanddealer,columns=['CardSlots'])
    rounddf['DealtCard1'] = list(starting_hand[::2])
    rounddf['DealtCard2'] = list(starting_hand[1::2])
    players_final_totals = []
    dealer_showing = rounddf.loc[rounddf['CardSlots']=='Dealer']['DealtCard1'].values[0]
    outcomes_df = pd.DataFrame(data=players, columns = ['Players'])
    player_outcomes =[]
    players_hands = []
    additional_cards = 0
    player_stops = []
    busted_list = []
    for player in players:
        blackjackcheck = False
        first_turn = True
        doubled = False
        players_current_hand = []
        players_current_hand.append(rounddf.loc[rounddf['CardSlots']==player]['DealtCard1'].values[0])
        players_current_hand.append(rounddf.loc[rounddf['CardSlots']==player]['DealtCard2'].values[0])
        player_current_total = get_hand_value(players_current_hand)
        extra_cards = [[] for player in players]
        players_final_totals = []

        blackjack_string_list = ['21','11/21']
        if player_current_total in blackjack_string_list:
            blackjackcheck = True

        if blackjackcheck == False:
            busted = False
            staying = False
            busted_or_staying = False
            double_list = []
            print()
            print(f"{player} it's your turn!")
            print()
            print(f'Dealer Showing: {dealer_showing}')
            print(f'Current Hand: {players_current_hand}')
            print(f'Current Total: {player_current_total}')
            while busted_or_staying == False:
                if first_turn == True:
                    possible_moves = ['hit', 'stay', 'double', 'split','im a bitch']
                    print('What would you like to do? \n To Hit, enter "hit" \n To Stay, enter "stay" \n To Double, enter "double" \n To Split, enter "split" \n To Surrender, enter "im a bitch"')
                else:
                    possible_moves = ['hit', 'stay']
                    print('What would you like to do? \n To Hit, enter "hit" \n To Stay, enter "stay"')
                
                move = 'temp'
                while move not in possible_moves:
                    move = input("What would you like to do")
                    
                print()
                print(f'{player} chooses to {move}')
                if move == "hit":
                    first_turn = False
                    newcard = draw_card()
                    newcardvalue = newcard['cards'][0]['value']
                    players_current_hand.append(newcardvalue)
                    player_current_total = get_hand_value(players_current_hand)

                    print(f'Card Dealt: {newcardvalue}')
                    print(f'Current Hand: {players_current_hand}')
                    print(f'Current Total: {player_current_total}')
                    if bustcheck(players_current_hand) == True:
                        print("That's Too Many!")
                        print("Busted")
                        doubled = False
                        double_list.append(doubled)
                        player_stops.append('Bust')
                        busted_list.append(busted)
                        players_final_totals.append(player_current_total)
                        busted = True
                        busted_or_staying = True
                elif move =="stay":
                    first_turn = False
                    players_final_totals.append(player_current_total)
                    player_stops.append("Stay")
                    busted_list.append(busted)
                    doubled = False
                    double_list.append(doubled)
                    staying = True
                    busted_or_staying = True
                elif move =="double":
                    first_turn = False
                    doubled = True
                    double_list.append(doubled)
                    newcard = draw_card()
                    newcardvalue = newcard['cards'][0]['value']
                    players_current_hand.append(newcardvalue)
                    player_current_total = get_hand_value(players_current_hand)
                    print(f'Card Dealt: {newcardvalue}')
                    print(f'Current Hand: {players_current_hand}')
                    print(f'Current Total: {player_current_total}')
                    if bustcheck(players_current_hand) == True:
                        print("That's Too Many!")
                        print("Busted")
                        player_stops.append('Bust')
                        busted_list.append(busted)                        
                        players_final_totals.append(player_current_total)
                        busted = True
                        busted_or_staying = True
                    staying = True
                    busted_or_staying = True      

        elif blackjackcheck == True:
            print()
            print(f"{player} it's your turn!")
            print()           
            print(f'Current Hand: {players_current_hand}')
            print(f'Current Total: {player_current_total}')
            print("Blackjack! (Pays 3:2)")
            print()
            busted_list.append(busted)
            player_stops.append("BlackJack")
        players_hands.append(players_current_hand)

    dealers_hand = []
    dealers_hand.append(rounddf.loc[rounddf['CardSlots']=="Dealer"]['DealtCard1'].values[0])
    dealers_hand.append(rounddf.loc[rounddf['CardSlots']=="Dealer"]['DealtCard2'].values[0])
    dealer_value = get_hand_value(dealers_hand) 
    dealer_check = does_dealer_hit(dealer_value)
    while dealer_check == True:
        print('Dealer Hits')
        dealercard = draw_card()
        dealercardvalue = dealercard['cards'][0]['value']
        dealers_hand.append(dealercardvalue)
        print(f'Dealer Draws: {dealercardvalue}')
        if bustcheck(dealers_hand) == True:
            print('Dealer Busts! Everybody wins')
            busted = True
            dealer_result = "Bust"
            busted_list.append(busted)
        dealer_value = get_hand_value(dealers_hand)
        print(f'Dealer Total: {dealer_value}')
        dealer_check = does_dealer_hit(dealer_value)
        if dealer_check == False:
            break

    double_list.append(np.nan)
    final_decisions = player_stops.copy()
    final_decisions.append(dealer_result)
    all_hands = players_hands.copy()
    all_hands.append(dealers_hand)

    rounddf['doubled'] = double_list
    rounddf['all_hands'] = all_hands
    rounddf['final_decisions'] = final_decisions

    return rounddf

def does_dealer_hit(dealer_value):
    if len(dealer_value)==2:
        if int(dealer_value)<17:
            return True
    else:
        if int(dealer_value[-2:]) in range(18,22):
            return False
        elif int(dealer_value.split('/')[0]) in range(17,22):
            return True
        else:
            return True
    if bustcheck(dealer_value) == True:
        return False


def bustcheck(hand):
    hand_string = get_hand_value(players_current_hand)
    if len(hand_string)<3:
        min_hand_value = int(hand_string)
    else:
        min_hand_value = int(hand_string[0:2])
    
    if min_hand_value > 21:
        return True
    else:
        return False




def get_hand_value(cards):
    players_current_hand=['ACE','7']
    number_list = ['1','2','3','4','5','6','7','8','9','10']
    face_list = 'JACK QUEEN KING'.split()
    total1 = 0
    total2 = 0
    acecheck = False
    totalstr = ''
    for card in cards:
        if card in number_list:
            cardval = int(card)
            total1=total1+cardval
            total2=total2+cardval
        elif card in face_list:
            cardval = 10
            total1 = total1+cardval
            total2 = total2 + cardval
        elif card == "ACE":
            total1 = total1 + 1
            total2 = total2 + 11
            acecheck = True
    if acecheck==False:
        totalstr = str(total1)
    else:
        totalstr = str(total1)+"/"+str(total2)
    return totalstr


        
        

        

def play_ball():
    print("Welcome to Blackjack, built by trevor ross")
    print()
    print()
    time.sleep(1)
    
    num_of_players = 6
    while num_of_players >= 5:
        try:
            num_of_players = int(input("How many players/hands should be dealt? (Maximum 4)"))
        except:
            time.sleep(0.25)
            print("Input must be a number")
            time.sleep(0.25)
    players_index = list(range(1,num_of_players+1))
    players_list = ["Player" + str(num) for num in list(range(1,num_of_players+1))]
    
    Players_df = pd.DataFrame(data = players_list,index=players_index,columns=['Names'])
    Players_df["Balance"] = [2000 for player in Players_df['Names']]

    allplayers = players_list.copy()
    allplayers.append("Dealer")
    Games_df = pd.DataFrame(data = allplayers,columns=['Players'])
    

    cd_list1 = [f'{player}Card1' for player in allplayers]
    cd_list2 = [f'{player}Card2' for player in allplayers]

    deal_cardsdf = pd.DataFrame()
    deal_cardsdf['Card1'] = cd_list1
    deal_cardsdf['Card2'] = cd_list2
    fullcdlist = []
    for i in range(0,len(cd_list2)):
        fullcdlist.append(cd_list1[i])
        fullcdlist.append(cd_list2[i])
    
    cards_dealt = pd.DataFrame(data=fullcdlist,columns=['CardSlots'])
    cards_dealt
    
    print("Each Player Has a Starting Balance of $2,000")
    time.sleep(0.3)
    balance_df = Players_df[["Names","Balance"]]
    balance_df["Money"] = [f'${balance}' for balance in balance_df['Balance']]
    balance_df.head()
    print("Current Balances")
    b = 1
    for player in players_list:
        pbal = balance_df['Money'][b]
        time.sleep(0.3)
        print(f'{player}: {pbal}')
    
    print()
    print()
    games = 1
    #Place your bets!
    balance_df[f'Wagers{games}'] = place_bets(players_list)



    #Deal Cards
    print("Dealing...")
    cards_dealt[f'Game{games}'] = deal_cards(deal_cardsdf,games)

    print()
    print()
    #Show the Board
    show_the_board(cards_dealt)

    play_a_round(players_list,cards_dealt)









#Images
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
img = mpimg.imread('/Users/tross/WILD/RossCo/Dev/deckofcards/static/img/KH.png')
imgplot = plt.imshow(img)
plt.show()