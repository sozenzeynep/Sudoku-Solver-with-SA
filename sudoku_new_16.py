import random
import numpy as np
import math
from random import choice
import statistics
from collections import OrderedDict
import matplotlib.pyplot as plt


count = 0

row_cost_dict = {}
column_cost_dict = {}

column_duplicates_dict = {}
row_duplicates_dict = {}
cost_list = []


startingSudoku = """
                    024007000
                    600000000
                    003680415
                    431005000
                    500000032
                    790000060
                    209710800
                    040093000
                    310004750
                """

aa = np.array([[int(i) for i in line] for line in startingSudoku.split()])

sudoku = np.array([
    [3, 13, 4, 0, 0, 9, 1, 0, 10, 8, 0, 2, 0, 0, 0, 5],
    [7, 0, 2, 0, 15, 0, 11, 4, 1, 0, 0, 0, 8, 14, 0, 0],
    [14, 8, 11, 6, 2, 7, 13, 12, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 12, 0, 0, 0, 0, 16, 3, 0, 0, 0, 0, 0, 9, 0, 0],
    [6, 0, 0, 0, 0, 15, 14, 0, 3, 0, 11, 10, 12, 0, 0, 0],
    [13, 0, 8, 0, 4, 0, 0, 5, 2, 0, 0, 0, 9, 6, 14, 11],
    [11, 0, 0, 12, 1, 0, 0, 0, 5, 0, 0, 0, 0, 16, 10, 0],
    [0, 2, 16, 15, 0, 0, 7, 0, 0, 0, 0, 12, 0, 8, 5, 0],
    [0, 1, 6, 0, 12, 0, 0, 0, 0, 2, 0, 0, 3, 10, 15, 0],
    [0, 9, 12, 0, 0, 0, 0, 16, 0, 0, 0, 5, 14, 0, 0, 6],
    [10, 16, 5, 3, 0, 0, 0, 7, 14, 0, 0, 4, 0, 2, 0, 8],
    [0, 0, 0, 14, 6, 10, 0, 11, 0, 15, 1, 0, 0, 0, 0, 16],
    [0, 0, 14, 0, 0, 0, 0, 0, 11, 10, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 16, 0, 0, 7, 1, 5, 14, 15, 11, 8, 13],
    [0, 0, 7, 13, 0, 0, 0, 1, 15, 12, 0, 16, 0, 5, 0, 14],
    [4, 0, 0, 0, 7, 0, 10, 15, 0, 6, 13, 0, 0, 12, 16, 9]
]
)


def PrintSudoku(sudoku):
    a = 12
    # print("\n")
    # for i in range(len(sudoku)):
    #     line = ""
    #     if i == 3 or i == 6:
    #         print("---------------------")
    #     for j in range(len(sudoku[i])):
    #         if j == 3 or j == 6:
    #             line += "| "
    #         line += str(sudoku[i, j]) + " "
    #     print(line)


def FixSudokuValues(fixed_sudoku):
    for i in range(0, 16):
        for j in range(0, 16):
            if fixed_sudoku[i, j] != 0:
                fixed_sudoku[i, j] = 1

    return (fixed_sudoku)


# Cost Function
def CalculateNumberOfErrors(sudoku):
    numberOfErrors = 0
    for i in range(0, 16):
        numberOfErrors += CalculateNumberOfErrorsRowColumn(i, i, sudoku)
    return (numberOfErrors)


def CalculateNumberOfErrorsRowColumn(row, column, sudoku):
    numberOfErrors = (16 - len(np.unique(sudoku[:, column]))) + (16 - len(np.unique(sudoku[row, :])))
    return (numberOfErrors)


def CalculateNumberOfErrorsColumn(column, sudoku):
    numberOfErrors = (16 - len(np.unique(sudoku[:, column])))
    return (numberOfErrors)


def CalculateNumberOfErrorsRow(row, sudoku):
    numberOfErrors = (16 - len(np.unique(sudoku[row, :])))
    return (numberOfErrors)


def CreateList3x3Blocks():
    finalListOfBlocks = []
    for r in range(0, 16):
        tmpList = []
        block1 = [i + 4 * ((r) % 4) for i in range(0, 4)]
        block2 = [i + 4 * math.trunc((r) / 4) for i in range(0, 4)]
        for x in block1:
            for y in block2:
                tmpList.append([x, y])
        finalListOfBlocks.append(tmpList)
    return (finalListOfBlocks)


def RandomlyFill3x3Blocks(sudoku, listOfBlocks):
    for block in listOfBlocks:
        for box in block:
            if sudoku[box[0], box[1]] == 0:
                currentBlock = sudoku[block[0][0]:(block[-1][0] + 1), block[0][1]:(block[-1][1] + 1)]
                sudoku[box[0], box[1]] = choice([i for i in range(1, 17) if i not in currentBlock])
    return sudoku


def SumOfOneBlock(sudoku, oneBlock):
    finalSum = 0
    for box in oneBlock:
        finalSum += sudoku[box[0], box[1]]
    return (finalSum)


def TwoRandomBoxesWithinBlock(fixedSudoku, block):
    while (1):
        firstBox = random.choice(block)
        secondBox = choice([box for box in block if box is not firstBox])

        if fixedSudoku[firstBox[0], firstBox[1]] != 1 and fixedSudoku[secondBox[0], secondBox[1]] != 1:
            return ([firstBox, secondBox])


def TwoRandomBoxesWithinBlock_New(fixedSudoku, block, firstBox_):
    while (1):
        firstBox = firstBox_
        secondBox = choice([box for box in block if box is not firstBox])
        while (secondBox[0] == firstBox[0] and secondBox[1] == firstBox[1]):
            secondBox = choice([box for box in block if box is not firstBox])

        if fixedSudoku[firstBox[0], firstBox[1]] != 1 and fixedSudoku[secondBox[0], secondBox[1]] != 1:
            return ([firstBox, secondBox])


def FlipBoxes(sudoku, boxesToFlip, type):
    try:
        # print('type ',type)
        proposedSudoku = np.copy(sudoku)
        placeHolder = proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]]
        proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]] = proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]]
        proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]] = placeHolder
        return (proposedSudoku)

    except:
        print("An exception occurred")

        a = 12
        return


def ProposedState_Init(sudoku, fixedSudoku, listOfBlocks):
    randomBlock = random.choice(listOfBlocks)

    # if SumOfOneBlock(fixedSudoku, randomBlock) > 6:
    #     return (sudoku, 1, 1)
    boxesToFlip = TwoRandomBoxesWithinBlock(fixedSudoku, randomBlock)
    proposedSudoku = FlipBoxes(sudoku, boxesToFlip, 1)
    return ([proposedSudoku, boxesToFlip])


def ProposedStateRow(sudoku, fixedSudoku, listOfBlocks, row_cost_dict):
    # randomBlock = random.choice(listOfBlocks)
    randomBlock = None
    sorted_row_cost_dict = OrderedDict(sorted(row_cost_dict.items(), key=lambda x: x[1]))
    aa = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15]
    b = random.choice(aa)
    nth_row_cost = list(sorted_row_cost_dict.items())[b]
    nth_row = sudoku[nth_row_cost[0], :]
    uniques, counts = np.unique(sudoku[nth_row_cost[0], :], return_counts=True)

    sort_index = np.argsort(counts)
    counts_temp = [counts[x] for x in sort_index]
    uniques_temp = [uniques[x] for x in sort_index]

    counts_temp.reverse()
    uniques_temp.reverse()

    found = False
    pair = []
    randomBlock = None
    if counts_temp[0] > 0:

        index = 0
        global_val = False
        boxes = None

        for unique in uniques_temp:
            potentials = [i for i, x in enumerate(nth_row.tolist()) if x == unique]
            for pot in potentials:
                selected_column_item = pot
                if fixedSudoku[nth_row_cost[0], selected_column_item] != 1:
                    pair = [nth_row_cost[0], selected_column_item]

                    for item in listOfBlocks:  # hangi
                        for sub_block in item:
                            if sub_block[0] == pair[0] and sub_block[1] == pair[1]:
                                randomBlock = item

                    # if SumOfOneBlock(fixedSudoku, randomBlock) > 6:
                    #     return (sudoku, 1, 1)

                    candidate = None
                    second_pair = []
                    for index in range(1, 17):
                        if index not in uniques:
                            candidate = index
                            for block in randomBlock:
                                if sudoku[block[0], block[1]] == candidate and fixedSudoku[block[0], block[1]] != 1:
                                    second_pair = block
                                    # if sudoku[pair[0], pair[1]] not in sudoku[:, second_pair[1]]:

                                    boxes = [pair, second_pair]
                                    proposedSudoku = FlipBoxes(sudoku, boxes, 2)

                                    currentCost = CalculateNumberOfErrors(sudoku)
                                    newCost = CalculateNumberOfErrors(proposedSudoku)

                                    if newCost < currentCost:
                                        global_val = True
                                        return ([proposedSudoku, boxes])
            else:
                continue
            break

        if global_val:
            return ([proposedSudoku, boxes])

    # candidate = None
    # second_pair = []
    # for block in randomBlock:
    #     if fixedSudoku[block[0], block[1]] != 1 and (block[0]!=pair[0] and block[1]!=pair[1]):
    #         # if sudoku[block[0], block[1]] not in sudoku[:, pair[1]]:
    #         second_pair = block

    return (sudoku, 1, 1)


def ProposedStateColumn(sudoku, fixedSudoku, listOfBlocks, column_cost_dict):
    # randomBlock = random.choice(listOfBlocks)
    randomBlock = None
    sorted_column_cost_dict = OrderedDict(sorted(column_cost_dict.items(), key=lambda x: x[1]))
    aa = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    b = random.choice(aa)
    nth_row_cost = list(sorted_column_cost_dict.items())[b]
    nth_row = sudoku[:, nth_row_cost[0]]
    uniques, counts = np.unique(sudoku[:, nth_row_cost[0]], return_counts=True)

    sort_index = np.argsort(counts)
    counts_temp = [counts[x] for x in sort_index]
    uniques_temp = [uniques[x] for x in sort_index]

    counts_temp.reverse()
    uniques_temp.reverse()

    found = False
    pair = []
    randomBlock = None
    if counts_temp[0] > 0:
        index = 0
        for unique in uniques_temp:
            potentials = [i for i, x in enumerate(nth_row.tolist()) if x == unique]
            for pot in potentials:
                selected_column_item = pot
                if fixedSudoku[selected_column_item, nth_row_cost[0]] != 1:
                    pair = [selected_column_item, nth_row_cost[0]]

                    for item in listOfBlocks:  # hangi
                        for sub_block in item:
                            if sub_block[0] == pair[0] and sub_block[1] == pair[1]:
                                randomBlock = item

                    if SumOfOneBlock(fixedSudoku, randomBlock) > 6:
                        return (sudoku, 1, 1)

                    candidate = None
                    second_pair = []
                    for index in range(1, 17):
                        if index not in uniques:
                            candidate = index
                            for block in randomBlock:
                                if sudoku[block[0], block[1]] == candidate and fixedSudoku[block[0], block[1]] != 1 and \
                                        pair[0] != block[0] and pair[1] != block[1]:
                                    second_pair = block
                                    if sudoku[pair[0], pair[1]] not in sudoku[second_pair[0], :]:
                                        boxes = [pair, second_pair]
                                        proposedSudoku = FlipBoxes(sudoku, boxes, 2)

                                        currentCost = CalculateNumberOfErrors(sudoku)
                                        newCost = CalculateNumberOfErrors(proposedSudoku)

                                        if newCost < currentCost:
                                            global_val = True
                                            return ([proposedSudoku, boxes])

            else:
                continue
            break

    # candidate = None
    # second_pair = []
    # for block in randomBlock:
    #     if fixedSudoku[block[0], block[1]] != 1 and (block[0]!=pair[0] and block[1]!=pair[1]):
    #         # if sudoku[block[0], block[1]] not in sudoku[:, pair[1]]:
    #         second_pair = block

    return (sudoku, 1, 1)


def ChooseNewState(currentSudoku, fixedSudoku, listOfBlocks, sigma, row_cost_dict, column_cost_dict):
    proposal = ProposedStateRow(currentSudoku, fixedSudoku, listOfBlocks, row_cost_dict)
    newSudoku = proposal[0]
    boxesToCheck = proposal[1]
    global count,cost_list
    # proposal = ProposedStateColumn(newSudoku, fixedSudoku, listOfBlocks, column_cost_dict)
    # newSudoku = proposal[0]
    # boxesToCheck = proposal[1]

    if type(boxesToCheck) is not list and boxesToCheck == 1:
        return ([currentSudoku, 0])

    # currentCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], currentSudoku)
    # currentCost = currentCost + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], currentSudoku)
    #
    # newCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], newSudoku)
    # newCost = newCost + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], newSudoku)

    currentCost = CalculateNumberOfErrors(currentSudoku)
    newCost = CalculateNumberOfErrors(newSudoku)
    costDifference = newCost - currentCost

    rho = math.exp(-costDifference / sigma)
    if (np.random.uniform(1, 0, 1) < rho):
        return ([newSudoku, costDifference])

    return ([currentSudoku, 0])


def ChooseNumberOfItterations(fixed_sudoku):
    numberOfItterations = 0
    for i in range(0, 16):
        for j in range(0, 16):
            if fixed_sudoku[i, j] != 0:
                numberOfItterations += 1
    return numberOfItterations


def CalculateInitialSigma(sudoku, fixedSudoku, listOfBlocks, row_cost_dict, column_cost_dict):
    listOfDifferences = []
    tmpSudoku = sudoku
    for i in range(1, 17):
        tmpSudoku = ProposedState_Init(tmpSudoku, fixedSudoku, listOfBlocks)[0]
        listOfDifferences.append(CalculateNumberOfErrors(tmpSudoku))
    return (statistics.pstdev(listOfDifferences))


def compute_row_and_column_cost(sudoku):
    # max_index = max(column_cost_dict, key=column_cost_dict.get)
    # from collections import OrderedDict
    # d_sorted_by_value = OrderedDict(sorted(column_cost_dict.items(), key=lambda x: x[1]))
    # nth_element = list(d_sorted_by_value.values())[8]

    for r in range(0, 16):
        row_cost_dict[r] = CalculateNumberOfErrorsRow(r, sudoku)
        column_cost_dict[r] = CalculateNumberOfErrorsColumn(r, sudoku)

    return row_cost_dict, column_cost_dict

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

# def solveSudoku(sudoku):
f = open("demofile2.txt", "a")
solutionFound = 0
tic()
while (solutionFound == 0):
    decreaseFactor = 0.99
    stuckCount = 0
    fixedSudoku = np.copy(sudoku)
    PrintSudoku(sudoku)
    FixSudokuValues(fixedSudoku)
    listOfBlocks = CreateList3x3Blocks()
    tmpSudoku = RandomlyFill3x3Blocks(sudoku, listOfBlocks)
    row_cost_dict, column_cost_dict = compute_row_and_column_cost(tmpSudoku)

    sigma = CalculateInitialSigma(sudoku, fixedSudoku, listOfBlocks, row_cost_dict, column_cost_dict)
    score = CalculateNumberOfErrors(tmpSudoku)
    itterations = ChooseNumberOfItterations(fixedSudoku)
    if score <= 0:
        solutionFound = 1

    while solutionFound == 0:
        previousScore = score
        for i in range(0, itterations):
            newState = ChooseNewState(tmpSudoku, fixedSudoku, listOfBlocks, sigma, row_cost_dict, column_cost_dict)

            tmpSudoku = newState[0]
            scoreDiff = newState[1]
            row_cost_dict, column_cost_dict = compute_row_and_column_cost(tmpSudoku)

            score += scoreDiff
            if score == 0:
                debug = 1
            count = count + 1;
            cost_list.append(score)

            if len(cost_list) > 9:
                found = True
                for i in range(len(cost_list) - 8, len(cost_list)):
                    if cost_list[len(cost_list) - 8] != cost_list[i]:
                        found = False

                if found:
                    toc()

                    debug = 1

            print(score,count)
            f.write(str(score) + '\n')
            if score <= 0:
                solutionFound = 1
                break

        sigma *= decreaseFactor
        if score <= 0:
            solutionFound = 1
            break
        if score >= previousScore:
            stuckCount += 1
        else:
            stuckCount = 0
        if (stuckCount > 80):
            sigma += 2
        if (CalculateNumberOfErrors(tmpSudoku) == 0):
            PrintSudoku(tmpSudoku)
            break
f.close()
# return (tmpSudoku)


# solution = solveSudoku(sudoku)
print(CalculateNumberOfErrors(tmpSudoku))
PrintSudoku(tmpSudoku)



plt.plot(cost_list)
plt.title('SUDOKU-MALİYET')
plt.ylabel('maliyet ')
plt.xlabel('iterasyon')
# plt.legend(['test'], loc='upper left')
plt.show()
