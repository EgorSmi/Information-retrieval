import re
import os
from collections import defaultdict
import codecs
import gzip
import struct
import sys
from copy import deepcopy
import operator
import pickle


class Fibonnachi:
    
    def __init__(self):
        self.fibonnachi_list = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                                987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 
                                121393, 196418, 317811, 514229, 832040]

    def encode(self, arr):
        value_list = ''
        flag = False
        for value in arr:
            value_list += '1'
            for k in self.fibonnachi_list[2:][::-1]:
                if k <= value:
                    value_list += '1'
                    flag = True
                    value -= k
                elif flag:
                    value_list += '0'
            flag = False
        return value_list
    
    def decode_elem(self, elem):
        value = 0
        for i in range(len(elem)):
            if elem[::-1][i] == '1':
                value += self.fibonnachi_list[2:][i]
        return value
    
    def decode(self, value_str):
        value_str = value_str[::-1]
        flag = False
        value_arr = []
        string=''
        for i in range(len(value_str)):
            if value_str[i] == '1':
                if flag:
                    value_arr.append(self.decode_elem(string))
                    flag = False
                    string = ''
                else:
                    flag = True
                    string = '1' + string
            else:
                string = '0' + string
                flag = False

        return value_arr[::-1]


class Indexing:
    def __init__(self):
        self.forward_index = defaultdict(list) # list of words
        self.backward_index = defaultdict(list) # list of doc_id
        self.doc_url = defaultdict(str)


    def build_forward_index(self, path_to_dataset):
        '''
        Построение прямого индекса по документам
        '''
        file_list = os.listdir(path_to_dataset)
        doc_id = 1
        for i in file_list:
            with gzip.open(path_to_dataset + '/' + i, 'rb') as f:
                while True:
                    first_byte = f.read(4)
                    if first_byte == b'':
                        break
                    size = struct.unpack('i', first_byte)[0]
                    text = f.read(size).decode('utf-8', 'ignore')
                    txt = re.split(r'\x1a{1,}', text)
                    url = txt[0]
                    self.doc_url[doc_id] = url[url.find('http'):]
                    tmp = []
                    for lines in txt[1:]:
                        tmp = re.findall(r'\w+', lines.lower())
                    self.forward_index[doc_id] = tmp
                    # периодически встречаются слова с дефисами... но решил их выкинуть)) под \w+ не подходят ведь)
                    doc_id += 1


    def build_backward_index(self):
        '''
        Построение обратного индекса. Просто берем слова без повторов из прямого индекса
        '''
        for key in self.forward_index.keys():
            for word in set(self.forward_index[key]):
                self.backward_index[word].append(key)

        self.backward_index_dict = deepcopy(self.backward_index)

    
    def forward_index_optimization(self, coder=Fibonnachi()):
        '''
        Оптимизируем прямой индекс. Каждому doc_id приписываем текст без повторяющихся в нем слов
        и кодируем с помощью codecs.encode()
        '''
        self.coder = coder
        for key in self.forward_index.keys():
            set_values = set(self.forward_index[key])
            self.forward_index[key] = codecs.encode(' '.join(list(set_values)))


    def backward_index_optimization(self, coder=Fibonnachi()):
        '''
        Терм в обратном индексе заменяем на его word_id, кодируем. Записываем posting list из промежутков
        между документами, кодируем posting list кодом Фибонначи
        '''
        self.coder = coder
        encoded_backward_index = {}
        cnt = 1 # счетчик обычный
        for key in self.backward_index.keys():
            encoded_backward_index[self.coder.encode([cnt])] = self.backward_index[key]
            cnt += 1
        self.backward_index = deepcopy(encoded_backward_index)
        encoded_backward_index.clear()

        for key in self.backward_index.keys():
            posting_list = []
            posting_list.append(self.backward_index[key][0])
            for i in range(1, len(self.backward_index[key])):
                posting_list.append(self.backward_index[key][i] - self.backward_index[key][i-1])
            self.backward_index[key] = self.coder.encode(posting_list)


    def make_dict(self):
        '''
        Для быстрого поиска создаем словарь с word_id, который считает как просто порядковый номер слова 
        в обратном индексе
        '''
        self.dict = {}
        cnt = 1 # счетчик обычный
        for key in self.backward_index_dict.keys():
            self.dict[key] = cnt
            cnt += 1
        self.backward_index_dict.clear()      


class Query_tree:
    def __init__(self):
        self.tree = {}


    def build(self, tokens, pos):
        '''
        Рекурсивно разбираем запрос и строим дерево запроса
        '''
        cnt = 0
        res = []
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == ')':
                cnt += 1
            elif tokens[i] == '(':
                cnt -= 1  
            elif tokens[i] == '|':
                res.append((i, cnt))
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == ')':
                cnt += 1
            elif tokens[i] == '(':
                cnt -= 1  
            elif tokens[i] == '&':
                res.append((i, cnt))
        # ищем минимум по cnt
        if len(res) != 0:
            res.sort(key = operator.itemgetter(1))
            self.tree[pos] = tokens[res[0][0]]
            self.build(tokens[:res[0][0]], 2*pos + 1)
            self.build(tokens[res[0][0] + 1:], 2*pos + 2)
        else:
            for i in range(len(tokens)):
                if tokens[i] == '!':
                    self.tree[pos] = tokens[i]
                    self.build(tokens[i+1:], 2*pos + 1)
                    break
                elif ((tokens[i] != '(') and (tokens[i] != ')')):
                    self.tree[pos] = tokens[i]


    def build_tree(self, query):
        '''
        Разделяем запрос на токены. Запускаем рекурсию
        '''
        tokens = re.findall(r'\w+|[\(\)&\|!]', query)
        self.build(tokens, 0)


    def build_posting_lists(self, index):

        for key in self.tree.keys():
            if ((self.tree[key] != '!') and (self.tree[key] != '|') 
                and (self.tree[key] != '&')):
                if self.tree[key] in index.dict.keys():

                    posting_list = index.coder.decode(index.backward_index[
                        index.coder.encode([index.dict[self.tree[key]]])])
                    
                    for i in range(1, len(posting_list)):
                        posting_list[i] = posting_list[i] + posting_list[i-1]   
                    self.tree[key] = posting_list
                else:
                    self.tree[key] = []

            
    def evaluate(self, pos):
        if self.tree[pos] == '&':
            left = self.evaluate(2*pos + 1)
            right = self.evaluate(2*pos + 2)
            return max(left, right)
        elif self.tree[pos] == '|':
            left = self.evaluate(2*pos + 1)
            right = self.evaluate(2*pos + 2)
            return min(left, right)
        elif self.tree[pos] == '!':
            val = self.evaluate(2*pos + 1)
            if val != self.doc_id:
                return self.doc_id
            else:
                return self.max_doc + 1
        if len(self.tree[pos]) != 0:
            return self.tree[pos][0]
        else:
            return self.max_doc + 1


    def goto(self, doc_id):
        for key in self.tree.keys():
            if ((self.tree[key] != '!') and (self.tree[key] != '|') 
                and (self.tree[key] != '&')):
                i = 0
                while ((i < len(self.tree[key])) and (self.tree[key][i] < doc_id)):
                    i += 1
                self.tree[key] = self.tree[key][i:]


    def return_urls(self, index_url):
        '''
        Потоковая обработка дерева
        '''
        self.url_id = []
        self.doc_id = 1
        self.max_doc = len(index.forward_index.keys())

        while self.doc_id <= self.max_doc:
            self.goto(self.doc_id)
            if self.doc_id == self.evaluate(0):
                self.url_id.append(self.doc_id)
            self.doc_id += 1

        urls = []

        for i in sorted(self.url_id):
            urls.append(index_url[i])
        
        return urls

with open('index.pkl', 'rb') as f:
    index = pickle.load(f)

input_str = sys.stdin.readlines()
# цикл по всем запросам
for query in input_str:
    query_tree = Query_tree()
    query_tree.build_tree(query.lower())
    query_tree.build_posting_lists(index)

    res = query_tree.return_urls(index.doc_url)
    print(query)
    print(len(res))
    for url in res:
        print(url)
