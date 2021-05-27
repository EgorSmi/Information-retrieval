import numpy as np
import gzip
import re
import sys
from collections import Counter, defaultdict
import codecs
import functools
import pickle
import heapq

class ErrorModel:
    def __init__(self):
        self.alpha = 2
        self.add_cnt = 0
        self.del_cnt = 0
        self.change_cnt = 0
        self.errors_cnt = 0
        self.stat = defaultdict(functools.partial(defaultdict, int)) # defaultdict(lambda : defaultdict(int))

    def make_stat(self, a, b):
        n, m = len(a), len(b)
        max_d = 4 * n * m
        matrix=self.lev_distance_matrix(a, b)
        dist = matrix[m, n]
        i, j = n, m
        while dist != 0: 
            actions =  [matrix[j - 1][i - 1] if (i > 0) and (j > 0) else max_d,  # change
                        matrix[j - 1][i] if j > 0 else max_d,  # add
                        matrix[j][i - 1] if i > 0 else max_d]  # delete
            action = np.argmin(actions)
            if action == 0:  # change
                if dist != actions[action]:
                    dist -= 1
                    self.stat[a[i-1]][b[j-1]] += 1
                    self.change_cnt += 1
                i -= 1
                j -= 1
            elif action == 1:  # add
                if dist != actions[action]:
                    dist -= 1
                    self.stat[''][b[j - 1]] += 1
                    self.add_cnt += 1
                j -= 1
            else:  # delete
                if dist != actions[action]:
                    dist -= 1
                    self.stat[a[i - 1]][''] += 1
                    self.del_cnt += 1
                i -= 1
    
    def calc_weights(self):
        self.errors_cnt = self.add_cnt + self.del_cnt + self.change_cnt
        self.weights = defaultdict(functools.partial(defaultdict, float))
        for first_letter, second_key in self.stat.items():
            for second_letter, cnt in second_key.items():
                self.weights[first_letter][second_letter] = -np.log(cnt / self.errors_cnt)


    def lev_distance_matrix(self, a, b):
        n, m = len(a), len(b)
        reverse = False
        if n > m:
            # убедимся что n <= m, чтобы использовать минимум памяти O(min(n, m))
            a, b = b, a
            n, m = m, n
            reverse = True
        current_row = list(range(n + 1))  # 0 ряд - просто восходящая последовательность (одни вставки)
        matrix = np.array([current_row])
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
            matrix = np.concatenate((matrix, [current_row]), axis=0)
        if reverse:
            return matrix.T
        else:
            return matrix


class LanguageModel:
    def __init__(self, unigrams, bigrams_cnt):
        self.unigram_stat = Counter(unigrams)
        self.bigram_stat = defaultdict(functools.partial(defaultdict, int)) # defaultdict(lambda : defaultdict(int))
        self.bigrams_cnt = bigrams_cnt
        
    def add_word_stat(self, word):
        self.unigram_stat[word] += 1
    
    def add_bigram_stat(self, bigram):
        words = bigram.split('|')
        self.bigram_stat[words[0]][words[1]] += 1
    
    def calc_weights(self):
        self.unigram_weights = defaultdict()
        self.bigram_weights = defaultdict(functools.partial(defaultdict, float))
        for word, cnt in self.unigram_stat.items():
            self.unigram_weights[word] = - np.log(cnt / len(self.unigram_stat.keys()))
            
        for first_key, second_key in self.bigram_stat.items():
            for key, cnt in second_key.items():
                self.bigram_weights[first_key][key] = - np.log(cnt / self.bigrams_cnt)


class Trie:
    def __init__(self, error_model, language_model):
        self.child = {}
        self.change_list = []
        self.error_model = error_model
        self.language_model = language_model
        self.limit_weight = 10
        self.rus_letters = list('йцукенгшщзхъфывапролджэёячсмитьбю')
        self.eng_letters = list("qwertyuiop[]asdfghjkl;'\zxcvbnm,.")
        
        
    def insert(self, word):
        current = self.child
        for l in word:
            if l not in current:
                current[l] = {}
            current = current[l]
        current['word'] = word
        current['language_weight'] = self.language_model.unigram_weights[word] # вес слова P(C)

    def search(self, word):
        current = self.child
        for l in word:
            if l not in current:
                return False
            current = current[l]
        if 'word' in current:
            return current['word']
        else:
            return False
    
    def make_trie(self, words):
        '''
        Берем все правильные слова и по ним строим бор
        '''
        uniq_words = set(words)
        for word in uniq_words:
            self.insert(word)

    
    def spellchecking(self, word, max_candidates=4):
        self.max_candidates = max_candidates
        self.change_list = []
        self.queue = []
        cnt_errors = 0
        self.rus_flag = False
        self.eng_flag = False
        heapq.heappush(self.queue, [word, '', self.child, 0, cnt_errors])
        while (len(self.queue) > 0):
            obj = heapq.heappop(self.queue)
            prefix = obj[0]
            word = obj[1]
            node = obj[2]
            weight = obj[3]
            cnt_errors = obj[4]
            if (len(prefix) == 0):
                # дошли до слова
                new = True
                if ('word' in node):
                    for obj in self.change_list:
                        if obj[0] == word:
                            obj[1] = min(weight, obj[1])
                            obj[3] = min(cnt_errors, obj[3])
                            new = False
                            break
                    if new:
                        self.change_list.append([word, weight, node['language_weight'], cnt_errors])
                continue

            for key in node:
                if (key in (self.rus_letters + self.eng_letters)) and (key != 'language_weight') and (key != 'word'):
                    # change nearest ab -> ba
                    if (len(prefix) > 1):
                        if (key == prefix[1] and prefix[0] in node[key]):
                            req_weight = 1.0 # штраф за перестановку букв
                            if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                                heapq.heappush(self.queue, [prefix[2:], word + prefix[1] + prefix[0], node[prefix[1]][prefix[0]], weight + req_weight, cnt_errors + 1])
                    if (prefix[0] in self.rus_letters) and (key == self.eng_letters[self.rus_letters.index(prefix[0])]):
                        req_weight = 0.1 # по сути это одна ошибка
                        if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                            heapq.heappush(self.queue, [prefix[1:], word + self.eng_letters[self.rus_letters.index(prefix[0])], node[key], weight + req_weight, cnt_errors + 1])
                    if (prefix[0] in self.eng_letters) and (key == self.rus_letters[self.eng_letters.index(prefix[0])]):
                        req_weight = 0.1 # по сути это одна ошибка
                        if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                            heapq.heappush(self.queue, [prefix[1:], word + self.rus_letters[self.eng_letters.index(prefix[0])], node[key], weight + req_weight, cnt_errors + 1])        


                    if key == prefix[0]:
                        # переход по нулевому весу
                        heapq.heappush(self.queue, [prefix[1:], word + prefix[0], node[key], weight, cnt_errors])
                        # нужно дублирование буквы cur_letter
                        if prefix[0] in self.error_model.weights['']:
                            req_weight = 0.2 * self.error_model.weights[''][prefix[0]]
                            if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                                heapq.heappush(self.queue, [prefix, word + prefix[0], node[prefix[0]], weight + req_weight, cnt_errors + 1])

                    else:
                        if key in self.error_model.weights[prefix[0]]:
                            # need change cur_letter -> key
                            req_weight = self.error_model.weights[prefix[0]][key]
                            if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                                heapq.heappush(self.queue, [prefix[1:], word + key, node[key], weight + req_weight, cnt_errors + 1])
                        if key in self.error_model.weights['']:
                            # miss letter key, need insert
                            req_weight = self.error_model.weights[''][key]
                            if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                                heapq.heappush(self.queue, [prefix, word + key, node[key], weight + req_weight, cnt_errors + 1])
            if '' in self.error_model.weights[prefix[0]]:
                # need to delete cur_letter
                req_weight = self.error_model.weights[prefix[0]]['']
                if (weight + req_weight < self.limit_weight and len(self.change_list) < self.max_candidates):
                    heapq.heappush(self.queue, [prefix[1:], word, node, weight + req_weight, cnt_errors + 1])
            continue

        return self.change_list


'''----------------Error model--------------------'''

er_model = ErrorModel()

with open("queries_all.txt", 'r') as file:
    lines = file.read().splitlines()

original_queries = []
fixed_queries = []
for line in lines:
    if '\t' in line:
        line = line.lower()
        original_queries.append(line[:(line.index('\t'))])
        fixed_queries.append(line[(line.index('\t')+1):])

for original, fixed in zip(original_queries, fixed_queries):
    er_model.make_stat(original, fixed)

er_model.calc_weights()

'''
with open('error_model.pkl', 'wb') as f:
    pickle.dump(er_model, f, protocol=2)
'''
p = pickle.Pickler(open('error_model.pkl', 'wb'))
p.fast = True
p.dump(er_model)

'''----------------Language model--------------------'''

all_words = []
for line in lines:
    words = re.findall(r'\w+', line.lower())
    all_words.extend(words) 

all_bi_grams = []
for line in lines:
    words = re.findall(r'\w+', line.lower())
    for i in range(len(words) - 1):
        bi_gram = words[i] + "|" + words[i+1]
        all_bi_grams.append(bi_gram)

lang_model = LanguageModel(all_words, len(all_bi_grams))

for bigram in all_bi_grams:
    lang_model.add_bigram_stat(bigram)

lang_model.calc_weights()

p = pickle.Pickler(open('lang_model.pkl', 'wb'))
p.fast = True
p.dump(lang_model)
'''
with open('lang_model.pkl', 'wb') as f:
    pickle.dump(lang_model, f, protocol=2)
'''