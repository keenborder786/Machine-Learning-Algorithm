15# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 23:11:53 2019

@author: MMOHTASHIM
"""

board=[" " for x in range(10)]


def insertLetter(letter,pos):
    board[pos]=letter
def spaceisFree(pos):
    return board[pos]==" "
def printBoard(board):
    print('  |   | ')
    print(""+board[1]+ " | "+ board[2]+ " | "+ board[3])
    print('  |   |')
    print("----------")
    print('  |   | ')
    print(""+board[4]+ " | "+ board[5]+ " | "+ board[6])
    print('  |   |')
    print("----------")
    print('  |   |')
    print(""+board[7]+ " | "+ board[8]+ " | "+ board[9])
    print('  |   |')
def isWinner(bo,le):##sorry for the long line
   return (bo[7]==le and bo[8]==le and bo[9]==le) or (bo[4]==le and bo[5]==le and bo[6]==le) or (bo[1]==le and bo[2]==le and bo[3]==le) or (bo[1]==le and bo[4]==le and bo[7]==le) or  (bo[2]==le and bo[5]==le and bo[8]==le) or  (bo[3]==le and bo[6]==le and bo[9]==le) or (bo[1]==le and bo[5]==le and bo[9]==le) or  (bo[3]==le and bo[5]==le and bo[7]==le)
def playerMove():
    run=True
    while run:
        move=input("please select a position to place an X (1-9): ")
        try:
            move=int(move)
            if move>0 and move<10:
                if spaceisFree(move):
                    run=False
                    insertLetter("X",move)
                else:
                    print("This space is occupied")
            else:
                print("Type a number witin the range")
        except:
            print("Type a number")
            
def compMove():
    possibleMoves=[x for x,letter in enumerate(board) if letter==" " and x!=0]
    move=0
    
    for let in ["O","X"]:
        for i in possibleMoves:
            boardCopy=board[:]
            boardCopy[i]=let
            if isWinner(boardCopy,let):
                move=i
                return move
    
    cornersOpen=[]
    for i in possibleMoves:
        if i in [1,3,7,9]:
            cornersOpen.append(i)
    if len(cornersOpen)>0:
        move=selectRandom(cornersOpen)
        return move
    
    if 5 in possibleMoves:
        move=5
        return move
    edgesOpen=[]
    for i in possibleMoves:
        if i in [2,4,6,8]:
            edgesOpen.append(i)
    if len(edgesOpen)>0:
        move=selectRandom(edgesOpen)
        return move

def selectRandom(li):
    import random
    In =len(li)
    r=random.randrange(0,In)
    return li[r]
def isBoardFull():
    if board.count(" ")>1:
        return False
    else:
        return True
def main():
    print("Welcome to Tic Tac Toe")
    printBoard(board)
    
    while not (isBoardFull()):
        if not isWinner(board,'O'):
            playerMove()
            printBoard(board)
        else:
            print("Sorry AI win the game")
            break
        if not isWinner(board,'X'):
            move=compMove()
            if move==0:
                print("Tie Game!")
            else:
                insertLetter("O",move)
                print("computer placed an O in position ,move")
                printBoard(board)
        else:
            print("Yeap You ARE smarter than my AI")
            break
    if isBoardFull():
        print("Tie Game")
main()