''' Для построения рекомендательной системы выбрана библиотека scikit-surprise 1.1.1. (surprise.readthedocs.io)
    Датасет данной библиотеки используют таблицу рейтингов из 3-х колонок user-item-rating.
    Для подготовки этих данных используется файл transactions.csv 
    Для расчета предсказаний выбран алгоритм cингулярного разложения SVD (Singular value decomposition)
    для экономии времени при повторных расчетах реализован механизм кеширования.
    После первоначального обучения алгоритма его расчетные данные записываются на диск (методом класса dump из этой же библиотеки).
    Повторный перерасчет алгоритма происходит если не найден его дамп или если расчетные данные устарели (при сравнении времени файла
     с временем файла транзакций). При расчете прогноза на больших входных данных (например по всем пользователям) используется 
     буферизованный вывод в файл. По умолчанию размер буфера установлен для расчета 100 пользователей (всего в датасете 100 000 пользователей). 

     Из дополнительных возможностей в тех. задании реализованы следующие функции:

    - По идентификатору пользователя выдавать набор из K наиболее релевантных для него товаров.
    - По массиву идентификаторов пользователей выдавать массив наборов из K наиболее релевантных для них товаров.
        При вызове функции predict() можно можно передавать id пользователей как строкой, так и списком.
        Например, predict(usr='7', k=10), predict(usr='10 22 42', k=15), predict(usr=['13', '22', '42'], k=4)
        В зависимости от типа параметра и количества пользователей система, вернет либо обычный одномерный список,
        рекомендуемых товаров, либо вложенный список для нескольких пользователей вида [['user_id', ['id1','id2'...]]]:
        [['13',['46', '117', '169', '264']],
         ['22',['695', '738', '1729', '1835']],
         ['42',['196', '377', '432', '1096']]]
        Если функцию predict() запустить без параметров, то она произведёт расчёт по всем пользователям и результаты
        запишет в файл submit.csv (в рабочем каталоге) в формате:
            user_id - уникальный идентификатор покупателя
            product_id - идентификаторы товаров через пробел в порядке убывания "уверенности модели"
    
    - Добавить свежие данные о транзакциях.
    - Обучить рекомендательную систему заново.
        Обновление данных происходит автоматически при (пере-)запуске системы в процедуре prepare_model().
        Обновление запуститься при отсутствии файлов сводного рейтинга rank.scv и файла кеша обученной модели algo.dmp.
    - Система уведомлений о состоянии алгоритма расчета и выполнении отдельных этапов. Уведомления можно отключать  .
    - Автоматическое возобновления прерванных длительных расчетов с позиции последнего сохранения буфера.
    - Реализован замер времени для отладки и оптимизации (его можно отключать).
    - В процедуре predict_extra реализован собственный скоростной алгоритм прогнозирования с использованием предварительно расчитанных данных.
'''

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import dump
import os
import csv
import time

# файлы данных, результаты и кеш
data_dir      = os.path.join(os.getcwd(), 'skillbox-recommender-system')
transact_file = os.path.join(data_dir, 'transactions.csv')
# файл рейтингов для загрузки в датасет модели
rank_file     = os.path.join(data_dir, 'rank.csv')
# algo_file - кеш обученной модели
algo_file     = os.path.join(data_dir, 'algo.dmp')
# status_file содержит id последнего сохраненного расчитанного (user_id) элемента
# При прерывании и возобновлении расчета позволяет сэкономить время
# и начать расчет сразу со следующего элемента
status_file   = os.path.join(data_dir, 'status.log')
# файл с результатами
result_file   = os.path.join(data_dir, 'submit2.csv')
ready_res_file= os.path.join(data_dir, 'submit_all_10.csv')
md_file       = os.path.join(data_dir, 'model_SVD.md')

print_log = True # выводить на экран процесс выполнения алгоритма
measure_time = True # замерять и выводить время работы
algo = [] # surprise.prediction_algorithms.matrix_factorization.SVD
# настройка buffer_size используется при полном расчете (для всех пользователей)
# для записи промежуточных результатов и экономии оперативной памяти
buffer_size = 100 # при достижении этого значения результаты дописываются в submit.csv

user_list = [] # все user_id, которые есть в транзакциях
item_list = [] # все item_id, которые есть в транзакциях
md_10 = {} # словарь {user_id: items_list} расчитанных предсказаний по 10 продуктам
md_all = [] # расчитанные предсказания по всем продуктам для новых пользователей

def predict_extra(usr='new', k=10):
    if not md_10:
        print('Ошибка: Не инициализированы данные md_10. Выполните сначала процедуру prepare_model или make_md.')
        return []
    if not md_all:
        print('Ошибка: Не инициализированы данные md_all. Выполните сначала процедуру make_md().')
        return []

    if len(usr):
        if type(usr) == str:
            if usr == 'new':
                return md_all
            else:    
                usr = usr.split(' ')
    else:
        return []       

    res = []
    for u in usr:
        if u in user_list:
            if k <= 10:
                res.append((u, list(md_10[u][:k])))
            else:
                r = list(md_10[u])
                id_all_list = 0
                for i in range(k-10):
                    while 1:
                        if md_all[id_all_list] in r:
                            id_all_list += 1
                        else:
                            break
                    r.append(md_all[id_all_list])
                    id_all_list += 1
                res.append((u, r))
        else:
            res.append((u, list(md_all[:max(k,len(md_all))])))
    return res


def make_md():
    global md_10, md_all
    if os.path.exists(md_file):
        (md_all, md_10), _ = dump.load(md_file)
        if print_log: print('log: Загружены md.')
        return
    if os.path.exists(ready_res_file):
        f = open(ready_res_file, 'r', encoding='UTF-8')
        txt = f.read()
        f.close()
        txt = txt.splitlines()
        md_10 = {}
        for s in txt[1:]:
            if s:
                user_id, item_id = s.split(',')
                item_id = item_id.split(' ')
                md_10[user_id] = item_id
    else:
        if print_log: print(f'log: Ошибка. Не найден файл: {result_file}')
        return

    #md_10 = tuple(md_10)
    test_set = []
    u = '1000000000000'
    for i in item_list:
        test_set.append((u, i, 0))
    predictions = algo.test(test_set)
    predictions.sort(key=lambda x: x.est, reverse=True)
    md_all = tuple(v.iid for v in predictions)
    dump.dump(md_file, predictions=(md_all, md_10))
    if print_log: print('log: Сформирован md.')

# Функция make_rank_file формирует file2.csv на основе группировки данных из file1
# по заданным колонкам cols, используя при парсинге разделители sep.
# Также имеется возможность пропустить заголовок в файле, если он есть (pass_header).
def make_rank_file(file1='', file2='', cols='1 6', sum_cols='', sep=',', pass_header=1):
    data_file = transact_file if not file1 else file1
    res_file = rank_file if not file2 else file2
    f = open(data_file, 'r', encoding='UTF-8')
    txt = f.read()
    f.close()
    txt = txt.splitlines()
    dic = {}
    col_list = list(map(int, cols.split()))
    sum_cols_list = list(map(int, sum_cols.split()))
    for i,line in enumerate(txt):
        if i < pass_header:
            continue
        line = line.strip()
        if not line:
            continue
        ln = line.split(sep)
        indx = ''
        for col in col_list:
            if indx:
                indx += '_'
            indx += str(ln[col])
        counter = 0
        counter = dic.setdefault(indx, 0)
        dic[indx] = counter + 1
    summedup_list = [ [*k.split('_'),v] for k,v in dic.items()]
    summedup_list.sort(key = lambda x: int(x[0]), reverse=False)
    with open(rank_file, 'w', encoding='UTF-8', newline='') as file:
        writer = csv.writer(file, delimiter=sep) # ;
        for item in summedup_list:
            writer.writerow(item)


def prepare_model():
    if not os.path.exists(status_file):
        f = open(status_file, 'w', encoding='UTF-8')
        f.write('done')
        #f.write('processing 75000')
        f.close()
    global algo, user_list, item_list
    if os.path.exists(algo_file):
        if print_log: print('log: Загружается модель...')
        (user_list, item_list), algo = dump.load(algo_file)
        if print_log: print('log: Модель загружена.')
        make_md()
        return True

    if not os.path.exists(rank_file):

        if not os.path.exists(transact_file):
            print('Выполнение прервано. Ошибка:')
            print(f' - в папке {data_dir} отсутствует (или не распакован) файл: transactions.csv')
            return False

        if print_log: print('log: Выполняется обработка транзакций (transactions.csv) ...')
        # обрабатываем файл транзакций, суммируя одинаковые значения
        # выбираем из него колонки 1 и 6 (user_id, product_id)
        # также указываем разделитель и пропуск строки с заголовком
        make_rank_file(cols='1 6', sep=',', pass_header=1)
        if print_log: print('log: Созданы рейтинги для обучения (rank.csv).')

    reader = Reader(line_format='user item rating', sep=',', skip_lines=0)
    data = Dataset.load_from_file(rank_file, reader=reader)
    #min_r, max_r = min(data.raw_ratings, key=lambda x: x[2]), max(data.raw_ratings, key=lambda x: x[2])
    #reader.rating_scale = min_r, max_r

    user_list = [e[0] for e in data.raw_ratings]
    user_list = list(set(user_list))
    user_list.sort(key = lambda x: int(x))
    item_list = [e[1] for e in data.raw_ratings]
    item_list = list(set(item_list))
    item_list.sort(key = lambda x: int(x))

    trainset = data.build_full_trainset()
    algo = SVD()
    if print_log: print('log: Выполняется обучение модели...')
    algo.fit(trainset)
    dump.dump(algo_file, algo=algo, predictions=(user_list, item_list))
    make_md()
    if print_log: print('log: Данные обученной модели сохранены в кеш (algo.dmp).')
    if print_log: print('log: Модель готова.')
    return True


def predict(usr=[], k=10, save_file=True):
    global algo
    if algo == []:
        print('Ошибка: Не инициализирована модель алгоритма. Выполните сначала процедуру prepare_model().')
        return

    if len(usr):
        if type(usr) == str:
            usr = usr.split(' ')
        save_file = False    
    else:
        usr = user_list        

    if print_log: print('log: Выполняется расчёт предсказаний...')
    if measure_time:
        time_start = time.time()
        time_start1 = time.time()
    buf_count = 0
    buffer = []
    stat_file = open(status_file, 'r')
    txt = stat_file.read()
    stat_file.close()
    if txt.find('processing') >= 0:
        cur_line = int(txt.split(' ')[1])
    else:
        cur_line = -1
    for indx, u in enumerate(usr):
        if cur_line >= indx:
            continue
        if indx == 0:
            with open(result_file, 'a') as res_file:
                res_file.writelines('user_id,product_id\n')
        test_set = []
        for i in item_list:
            test_set.append((u, i, 0))
        predictions = algo.test(test_set)
        predictions.sort(key=lambda x: x.est, reverse=True)
        buffer.append((u, [v.iid for v in predictions[:k]]))
        buf_count += 1
        if buf_count >= buffer_size and save_file:
            txt = ''
            for b in buffer:
                txt += b[0] + ',' + ' '.join(b[1]) + '\n'
            with open(result_file, 'a') as res_file:
                res_file.writelines(txt)
            with open(status_file, 'w') as stat_file:
                stat_file.write(f'processing {indx}')
            if print_log: print(f'log: Выполнено {indx}.')
            buf_count = 0
            buffer = []
            if measure_time:
                time_end1 = time.time()
                print('Время {:.2f} сек'.format(time_end1-time_start1))
                time_start1 = time.time()

    if buffer and save_file:
        txt = ''
        for b in buffer:
            txt += b[0] + ',' + ' '.join(b[1]) + '\n'
        with open(result_file, 'a') as res_file:
            res_file.writelines(txt)
        with open(status_file, 'w') as stat_file:
            stat_file.write('done')
        buf_count = 0
        buffer = []    
    if print_log: print('log: Расчет выполнен.')
    if measure_time:
        time_end = time.time()
        print('Время {:.2f} сек'.format(time_end-time_start))

    if not save_file:
        if len(buffer) == 1:
            return buffer[0][1]
        elif len(buffer) > 1:
            return buffer

    
if __name__ == '__main__':
    prepare_model()
    # if prepare_model():
    #     predict()