# python3 -m pip install num2words

from num2words import num2words

def gen_math_text():
    # number to words
    for i in range(1, 100):
        print(f"{i} = {num2words(i)}")

    # words to number
    for i in range(1, 100):
        print(f"{num2words(i)} = {i}")

    # Additions
    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{i} + {j} = {i + j}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} plus {num2words(j)} = to {num2words(i + j)}")


    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} + {num2words(j)} == to {num2words(i + j)}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} plus {num2words(j)} is equal to {num2words(i + j)}")


    # Subtractions
    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{i} - {j} = {i - j}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} minus {num2words(j)} = to {num2words(i - j)}")


    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} - {num2words(j)} == to {num2words(i - j)}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} - {num2words(j)} is equal to {num2words(i - j)}")

    # Multiplications

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{i} * {j} = {i * j}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} * {num2words(j)} = to {num2words(i * j)}")


    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} times {num2words(j)} == to {num2words(i * j)}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} * {num2words(j)} is equal to {num2words(i * j)}")

    # Divisions

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{i} / {j} = {(i / j).__round__(2)}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} / {num2words(j)} = to {num2words((i / j).__round__(2))}")


    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} divided by {num2words(j)} == to {num2words((i / j).__round__(2))}")

    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{num2words(i)} / {num2words(j)} is equal to {num2words((i / j).__round__(2))}")


gen_math_text()
