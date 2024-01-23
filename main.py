import numpy as np
import matplotlib.pyplot as plt
import argparse




def openfile(path):
    try:
        file = open(path, "r", encoding ="utf-8")
        storyStr = file.read()
        return storyStr
    except Exception as e:
        print(f"Error with file path:{e}")




#for this function i had to look it up for the lowercase ndarray lower funciton:https://numpy.org/doc/stable/reference/generated/numpy.char.lower.html
def lower(inStr):
    return np.char.lower(inStr)

def tokenization(inStr):
    dumpOut = ""
    for character in inStr:
        if character.isalpha() or character.isspace():
            dumpOut = dumpOut + character
        else:
            dumpOut = dumpOut + " "

    inStr = dumpOut
    inStr = inStr.replace("\n", " ")
    inStr = inStr.split(" ")

    tokens = np.array(inStr, dtype=str)
    vocab, counts = np.unique(tokens, return_counts = True)
    print(f"Number of tokens found: {len(tokens)}")
    return tokens

def stemming(tokens):
    #I did us chat gpt to write this list by asking it to remove all non suffix (same with the preffix list) material from the list
    #I also manually added some to the list below that I didn't see from the list i found
    #resource used https://www.fldoe.org/core/fileparse.php/16294/urlt/morphemeML.pdf

    suffix_list = [
        "er", "ly", "able", "ible", "hood", "ful", "less",
        "ish", "ness", "ic", "ist", "ian", "or", "eer", "ology",
        "ship", "ous", "ive", "age", "ant", "ant", "ent", "ent",
        "ment", "ary", "ize", "ise", "ure", "ion", "s", "ed",
        "ation", "ance", "ence", "ity", "al", "al", "al",
        "ate", "tude", "ism", "ing",
    ]

    prefix_list = [
        "de", "dis", "trans", "dia", "ex", "e", "mono", "uni",
        "bi", "di", "tri", "multi", "poly", "pre", "post", "mal",
        "mis", "bene", "pro", "sub", "re", "inter", "intra", "co",
        "com", "con", "col", "be", "non", "un", "in", "im", "il",
        "ir", "in",
        "a", "an", "anti", "contra", "counter", "en", "em",
        "astr", "bi", "geo", "therm", "auto", "homo", "hydro", "micro",
        "macro"
    ]

    dump = []
    for tok in tokens:
        print(tok)
        suf_flag = False
        pre_flag = False

        for i in suffix_list:
            if tok[len(i):] == i:
                if suf_flag == False:
                    tok = tok[:-len(i)]
                    suf_flag = True


        for k in prefix_list:
            if tok[:len(k)] == k:
                if pre_flag == False:
                    tok = tok[len(k):]
                    pre_flag = True
        print(tok)
        if tok != "":
            dump.append(tok)
    print(dump)
    return np.array(dump)


def stopwords(tokens):

    #again i used chat gpt to quicly convert the listed stopwords from https://productresources.collibra.com/docs/collibra/latest/Content/DGCSettings/ServicesConfiguration/co_stop-words.htm#:~:text=The%20list%20typically%20contains%20articles,%2C%20was%2C%20will%20and%20with.
    # to a python list(promt: a, an, and, are, as, at, be, but, by, for, if, in, into, is, it, no, not, of, on, or, such, that, the, their, then, there, these, they, this, to, was, will and with make this a python list)
    #I also manually add " " and ""
    common_stopwords = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "s"
        "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "her",
        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", " ", "", "i", "you"
    ]
    dumptokens = []
    for m in tokens:
        if m not in common_stopwords:
            dumptokens.append(m)

    return np.array(dumptokens)

def shortwords(tokens):
    dump = []
    for m in tokens:
        if len(m) >= 3 :
            dump.append(m)
    return np.array(dump)

def visualization(tokens):
    # I only show the top 100 tokens with the frequency of amount of tokens for more tokens you can manually change below
    sizeOf = 100
    vocab, counts = np.unique(tokens, return_counts=True)
    print(f"After preprocessing there are {tokens.size} tokens & {vocab.size} vocab")

    #I asked chatgpt for help on ordering the list so the next line is the generated sorted list indices which contain the indices of the counts from max to min
    sorted_indices = np.argsort(counts)[::-1]
    ### end of help ###
    print("This will take a min")
    for i in sorted_indices:
        print(f"{vocab[i], counts[i]}")

    plt.figure(figsize=(18,10))
    plt.bar(vocab[sorted_indices], counts[sorted_indices])
    plt.yscale("log")
    plt.xlabel("Token")
    plt.ylabel("Counts")
    plt.title("Tokens and associated Counts")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
story = openfile("bibleJMS.txt")
visualization(stopwords(shortwords(stemming(lower(tokenization(story))))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tokenization, preprocessing, and visualization")

    parser.add_argument("-p", "--path", type=str, help="Full filepath to txt file for preprocessing")
    parser.add_argument("--lower", action="store_true", help="perform lowercasing on text")
    parser.add_argument("--stemming", action="store_true", help="perform lowercasing on text")
    parser.add_argument("--shortwords", action="store_true", help="perform lowercasing on text")
    parser.add_argument("--stopwords", action="store_true", help="perform lowercasing on text")
    parser.add_argument("--visualization", action="store_true", help="perform lowercasing on text")

    arguments = parser.parse_args()

    path = arguments.path or "bibleJMS.txt"

    story = openfile(path)
    tokens = tokenization(story)

    if arguments.lower == True:
        tokens = lower(tokens)
    if arguments.stemming == True:
        tokens = stemming(tokens)
    if arguments.shortwords == True:
        tokens = shortwords(tokens)
    if arguments.stopwords == True:
        tokens = stopwords(tokens)
    if arguments.visualization == True:
        visualization(tokens)





