import json
from tqdm import tqdm

text = """
In the above example, the output of the BPE is a vocabulary,
which can be used to encode any text that is written with the letters "abcd". 
It will not be able to encode text containing other symbols, such as "no". 
Even giving each of the 26 letters an entry in the vocabulary, 
since there are many languages in the world using many different scripts, 
inevitably some symbols would be unencodable by such a vocabulary.
"""
class Tokenizer:
    def __init__(self,vocab_size,vocab_dict = {}, save_path=None):
        self.vocab_size = vocab_size
        self.vocab_dict = vocab_dict
        self.save_path = save_path

    def token_counter(self,bytes_txt):
        countTab = {}
        for i, j in zip(bytes_txt, bytes_txt[1:]):
            countTab[(i, j)] = countTab.get((i, j), 0) + 1
        return countTab

    def getMax(self,bytes_txt):
        countTab = self.token_counter(bytes_txt)
        return max(countTab, key=lambda v: countTab[v], default=None)

    def replaceMaxTok(self,bytes_txt, repPair, repid):
        i = 0
        new_bytes = []
        while i < len(bytes_txt):
            if i < len(bytes_txt) - 1 and (bytes_txt[i], bytes_txt[i+1]) == repPair:
                new_bytes.append(repid)
                i += 2
            else:
                new_bytes.append(bytes_txt[i])
                i += 1
        return new_bytes

    def replaceIdByPair(self,bytes_txt, id_, pair):
        i = 0
        new_bytes = []
        while i < len(bytes_txt):
            if bytes_txt[i] == id_:
                new_bytes.extend(pair)
            else:
                new_bytes.append(bytes_txt[i])
            i += 1
        return new_bytes

    def trainTokenizer(self,training_string):
        bytes_txt = list(training_string.encode())
        

        for i in tqdm(range(256, self.vocab_size + 256)):
            maxPair = self.getMax(bytes_txt)
            if not maxPair: break
            self.vocab_dict[i] = maxPair
            bytes_txt = self.replaceMaxTok(bytes_txt, maxPair, i)
        if self.savePath:
            self.saveTokenizer(self.savePath)
        return self.vocab_dict
    
    def loadFromSave(self,path):
        with open(path, "r") as f:
            self.vocab_dict = json.loads(f.read())

    def saveTokenizer(self, path):
        with open(path, "w+") as f:
            f.write(json.dumps(self.vocab_dict))

    def encoder(self,str_: str):
        bytes_txt = list(str_.encode())
        for k, v in sorted(self.vocab_dict.items()):
            bytes_txt = self.replaceMaxTok(bytes_txt, v, k)
        return bytes_txt
    

    def decoder(self,token_list: list[int]):
        bytes_txt = token_list[:]
        for k, v in sorted(self.vocab_dict.items(), reverse=True):
            bytes_txt = self.replaceIdByPair(bytes_txt, k, v)
        return bytes(bytes_txt).decode('utf-8', errors='replace')
