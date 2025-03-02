import m1
import os
import json


def read_file(file_input):
        file_size = os.path.getsize(file_input)
        stop_words = ('a','about','above','after','again','against','all','am','an',"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've", "your",'yours','yourself','yourselves')
        with open(file_input, 'r') as file:
            data = json.load(file)
        total_docs = data['total_docs']
        data_list_words = list(data['doc_freq'])
        count = 0
        for words in data_list_words:
            if(words not in stop_words):
                count += 1
        print('Total Number of Docs:',total_docs)
        print("Number of Unique words:",count)
        print("The size of the file is: ", file_size/1024, "KB")

if __name__ == "__main__":
    read_file("inverted_index.json")
