import pickle
import operator
import numpy as np
from collections import Counter, defaultdict
import re
import functools
import pickle
import heapq

class LanguageModel:
    def __init__(self, unigrams, bigrams_cnt):
        self.unigram_stat = Counter(unigrams)
        self.bigram_stat = defaultdict(functools.partial(defaultdict, int)) #defaultdict(lambda : defaultdict(int))
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

class ErrorModel:
    def __init__(self):
        self.alpha = 2
        self.add_cnt = 0
        self.del_cnt = 0
        self.change_cnt = 0
        self.errors_cnt = 0
        self.stat = defaultdict(functools.partial(defaultdict, int)) #defaultdict(lambda : defaultdict(int))

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

class Trie:
    def __init__(self, error_model, language_model):
        self.child = {}
        self.change_list = []
        self.error_model = error_model
        self.language_model = language_model
        self.limit_weight = 11
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

class Workflow:
    def __init__(self, trie, unigram_stat, bigram_stat, bigrams_cnt):
        self.trie = trie
        self.unigram_stat = unigram_stat
        self.bigram_stat = bigram_stat
        self.bigrams_cnt = bigrams_cnt


    def count_proba(self, request):
        '''
        Смотрим, насколько запрос реален
        Доп. штраф на слова длиной 1-3 буквы
        '''
        words = request.split()
        delta = 10e-12
        mas = []
        if len(words) > 1:
            for word in words:
                k = 1
                if len(word) == 1:
                    k = 1.7
                elif len(word) == 2:
                    k = 1.3
                elif len(word) > 12:
                    k = 3.5
                mas.append(-np.log((self.unigram_stat[word] + delta) / (len(self.unigram_stat.keys()) * (1 + delta))) * k)
        else:
            # особая магия для запросов из одного слова
            if request != '':
                mas.append(-np.log((self.unigram_stat[request] + delta) / (len(self.unigram_stat.keys()) * (1 + delta))) * 0.1 * len(request))
            else:
                mas.append(100)
        prob = np.sum(np.asarray(mas))
        # что-то вроде сглаживания Лапласа
        return prob


    def bigram_proba(self, request):
        '''
        Смотрим, насколько запрос реален по биграммным статистикам.
        Работает хуже, чем обычный count_proba
        '''
        words = request.split()
        delta = 10e-7
        if len(words) > 1:
            prob = sum([-np.log((self.bigram_stat[words[i]][words[i-1]] + delta) / (self.bigrams_cnt * (1 + delta)))
                        for i in range(1, len(words))])
        else:
            prob = -np.log(delta / (self.bigrams_cnt * (1 + delta)))
        return prob


    def get_proba(self, request):
        '''
        Смотрим, насколько запрос реален по биграммным и униграммным статистикам.
        Но, честно говоря, мне не нравится результат из-за того, что статистики не много
        '''
        words = request.split()
        delta_bi = 10e-7
        delta_uni = 10e-12
        res = []
        if len(words) > 1:
            for i in range(1, len(words)):
                k = 1
                if len(words[i]) == 1:
                    k = 1.7
                elif len(words[i]) == 2:
                    k = 1.3
                elif len(words[i]) > 12:
                    k = 3.5
                if words[i] in self.bigram_stat and words[i-1] in self.bigram_stat[words[i]]:
                    res.append(-np.log((self.bigram_stat[words[i]][words[i-1]] + delta_bi) / (self.bigrams_cnt * (1 + delta_bi))) * k)
                else:
                    res.append(-np.log((self.unigram_stat[words[i]] + delta_uni) / (len(self.unigram_stat.keys()) * (1 + delta_uni))) * k)
        else:
            if request != '':
                res.append(-np.log((self.unigram_stat[request] + delta_uni) / (len(self.unigram_stat.keys()) * (1 + delta_uni))) * 0.1 * len(request))
            else:
                res.append(100)
        return np.sum(np.asarray(res))


    def layout_classifier(self, request, proba):
        '''
        Классификатор раскладки
        '''
        layout_req = self.layout_generator(request)
        cur_proba = proba(request)
        layout_proba = proba(layout_req) 
        if cur_proba > layout_proba:
            return layout_req, layout_proba
        return request, cur_proba


    def join_generator(self, request, proba):
        '''
        Перебор по всем пробелам. O(кол-во пробелов)
        '''
        indices = [i for i in range(len(request)) if request[i] == ' ']
        new_req = request
        new_proba = 100 # magic const
        tmp_cnt = 0
        for indx in indices:
            tmp = new_req[:indx - tmp_cnt] + new_req[indx - tmp_cnt + 1:]
            tmp_proba = proba(tmp)
            if tmp_proba < new_proba:
                new_req = tmp
                new_proba = tmp_proba
                tmp_cnt += 1
        return new_req, new_proba


    def split_generator(self, request, proba):
        '''
        Перебор по всем буквам. O(кол-во букв)
        Нужен доп штраф на слова длиной 1-3 буквы
        '''
        indices = [i for i in range(1, len(request)) if request[i] != ' ']
        new_req = request
        new_proba = 100 # magic const
        tmp_cnt = 0
        for indx in indices:
            tmp = new_req[:indx + tmp_cnt] + ' ' + new_req[indx + tmp_cnt:]
            tmp_proba = proba(tmp)
            req_words = new_req.split()
            limit_value = 4 * len(req_words)
            if tmp_proba < new_proba and tmp_proba > limit_value:
                new_req = tmp
                new_proba = tmp_proba
                tmp_cnt += 1
        return new_req, new_proba


    def word_generator(self, request, max_candidates=4):
        '''
        Исправляем опечатки в словах запроса
        '''
        words = re.findall(r'\w+', request)

        for word in words:
            change_list = self.trie.spellchecking(word, max_candidates)
            if len(change_list) != 0:
                yield sorted([(obj[0], obj[1], obj[2], obj[3], obj[1] + obj[2]) for obj in change_list], 
                                        key=operator.itemgetter(4))[:self.trie.max_candidates]
            else:
                yield []
    
    
    def word_classifier(self, request, max_candidates=4):
        '''
        Алгоритм: ищем лучшие пары по биграммной статистики.. ну и магии
        '''
        fix_words = list(self.word_generator(request, max_candidates))
        # fix_words - [ [ (.....), (.....), (.....), (.....) ], [ (.....), (.....), (.....), (.....) ] ... ]
        if len(fix_words) > 1:
            # at least two words
            words = request.split()
            best_req = words[0]
            best_score = 100 # magic const
            best_cand = words[1] # word
            if len(fix_words[0]) != 0:
                for first in fix_words[0]:
                    if len(fix_words[1]) != 0:
                        for second in fix_words[1]:
                            tmp = first[0] + ' ' + second[0]
                            tmp_score = self.get_proba(tmp) + 2.7 * first[4] + 2.7 * second[4]
                            if tmp_score < best_score:
                                best_req = tmp
                                best_cand = second[0]
                                best_score = tmp_score
                    else:
                        tmp = first[0] + ' ' + words[1]
                        tmp_score = self.get_proba(tmp) + 2.7 * first[4]
                        if tmp_score < best_score:
                            best_req = tmp
                            best_cand = words[1]
                            best_score = tmp_score
                result_req = best_req
                first = best_cand  # word 
            else:
                result_req = words[0]
                if len(fix_words[1]) != 0:
                    for second in fix_words[1]:
                        tmp = best_req + ' ' + second[0]
                        tmp_score = self.get_proba(tmp) + 2.7 * second[4]
                        if tmp_score < best_score:
                            best_req = tmp
                            best_cand = second[0]
                            best_score = tmp_score
                    result_req = best_req
                    first = best_cand  # word 
                else:
                    first = words[1]
                    result_req += ' ' + first
            for i in range(2, len(fix_words)):
                best_req = request
                best_score = 100 # magic const
                if len(fix_words[i]) != 0:
                    for second in fix_words[i]:
                        tmp = first + ' ' + second[0]
                        tmp_score = self.get_proba(tmp) + 2.7 * second[4]
                        if tmp_score < best_score:
                            best_req = tmp
                            best_cand = second[0]
                            best_score = tmp_score
                    result_req += ' ' + best_cand
                    first = best_cand  
                else:
                    result_req += ' ' + words[i]
                    first = words[i]
            return result_req
        else:
            if len(fix_words) != 0:
                if len(fix_words[0]) > 0:
                    return fix_words[0][0][0]
            return ''
        
    
    def work(self, request, proba, iterations=2):
        '''
        Iterations
        '''
        for i in range(iterations):
            request2, it2_proba = self.split_generator(request, proba)
            request3, it3_proba = self.join_generator(request, proba)
            request1 = self.word_classifier(request, iterations - i)
            it1_proba = proba(request1)
            split_fix = it1_proba - it2_proba
            join_fix = it1_proba - it3_proba
            magic_const = 3.0
            if (split_fix > 0.0) and (split_fix < magic_const): # magic const....
                # выбираем fix
                request = request1
            elif (join_fix > 0.0) and (join_fix < magic_const):
                # выбираем fix
                request = request1
            else:
                # выбираем минимум по весу
                if it2_proba < it1_proba:
                    if it2_proba < it3_proba:
                        request = request2
                    else:
                        request = request3
                else:
                    if it1_proba < it3_proba:
                        request = request1
                    else:
                        request = request3
        return request
    
    
    def layout_generator(self, request):
        '''
        Генератор раскладки
        '''
        self.rus_letters = list('йцукенгшщзхъфывапролджэёячсмитьбю')
        self.eng_letters = list("qwertyuiop[]asdfghjkl;'\zxcvbnm,.")
        layout_req = ""
        for letter in request:
            if letter in self.rus_letters:
                layout_req += self.eng_letters[self.rus_letters.index(letter)]
            elif letter in self.eng_letters:
                layout_req += self.rus_letters[self.eng_letters.index(letter)]
            else:
                layout_req += letter
        return layout_req


with open('lang_model.pkl', 'rb') as f:
    lang_model = pickle.load(f)

with open('error_model.pkl', 'rb') as f:
    er_model = pickle.load(f)

with open("queries_all.txt", 'r') as file:
    lines = file.read().splitlines()

correct_words = []
for line in lines:
    if '\t' in line:
        line = line[(line.index('\t')+1):]
    line = re.findall(r'\w+', line.lower())
    correct_words.extend(line)


trie = Trie(er_model, lang_model)

trie.make_trie(correct_words)

wf = Workflow(trie, lang_model.unigram_stat, lang_model.bigram_stat, lang_model.bigrams_cnt)


while (True):
    try:
        query = input()
    except EOFError:
        break
    print(wf.work(query, proba=wf.count_proba)) 