def find_substring_position(main_string, substring,LABEL):
    start = main_string.find(substring)
    if start == -1:
        return -1, -1  # 如果子字符串不在主字符串中，则返回(-1, -1)
    end = start + len(substring)

    return (start,end,LABEL)
test_data = [
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 30, "GPE"), (44, 52, "MONEY")]}),
    ("Elon Musk is the CEO of SpaceX and Tesla.", {"entities": [(0, 9, "PERSON"), (23, 28, "ORG"), (33, 38, "ORG")]}),
]


persons = ["Jay Zhou","Bruce Li"]
amounts = ["$1.1","one dollar","10 dollars"]
actions = ["I want to transfer","Send"]
dates = ["Today","this week","2024-04-29","last 7 days"]

def build_test_data():
    for action in actions:
        for amount in amounts:
            for person in persons:
                for date in dates:
                    entities = []
                    text = f"{action} {amount} to {person} {date}"
                    entities.append(find_substring_position(text,amount,"AMOUNT"))
                    entities.append(find_substring_position(text,person,"PERSON"))
                    entities.append(find_substring_position(text,date,"DATE"))
                    test_data.append((text,{"entities":entities}))

if __name__ == '__main__':

    # text = "I want to transfer $1 to joey on 2024-04-29"
    # date = "2024-04-29"
    # entitly= find_substring_position(text, date,"DATE")
    # print(entitly)
    build_test_data()
    for test_datum in test_data:
        print(test_datum)
