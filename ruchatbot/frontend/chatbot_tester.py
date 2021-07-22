# -*- coding: utf-8 -*-
"""
Автоматическое тестирование диалогов, сценариев, форм в чатботе https://github.com/Koziev/chatbot
См. для справки "A User Simulator for Task-Completion Dialogues" https://arxiv.org/pdf/1612.05688.pdf
16.07.2020 Начальная версия
18.07.2020 Добавлены правила с регулярками для реакции на B-фразы
25.10.2020 Добавлен коннектор с сервисом ruGPT-читчата
07.11.2020 Добавлены фразы прощания в ситуации, когда бот остановил выдачу реплик
10.01.2021 Добавлена опция комстроки --user_id для задания идентификатора собеседника при проведения нагрузочного тестирования
"""

import os
import argparse
import logging
import io
import random
import collections
import re

import numpy as np
import requests

import rutokenizer
from ruchatbot.bot.console_utils import flush_logging
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.frontend.bot_creator import create_chatbot, ChitchatConfig
#from ruchatbot.gpt2.gpt_chitchat import Gpt2Chitchat
from ruchatbot.gpt2.rugpt2_chitchat_connector import RuGpt2ChitchatConnector


def ngrams(s, n):
    # return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len=3):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-6+len(shingles1 | shingles2))


def one_of(needle, hay):
    for hay_item in hay:
        s = jaccard(needle, hay_item)
        if s >= 0.90:
            return hay_item

    return None


def one_of2(needle, hay):
    for hay_item in hay:
        s1 = jaccard(needle[0], hay_item[0])
        s2 = jaccard(needle[1], hay_item[1])
        s = s1 * s2
        if s >= 0.80:
            return hay_item

    return None


def on_order(order_anchor_str, bot, session):
    bot.say(session, u'Выполняю команду \"{}\"'.format(order_anchor_str))
    # Всегда возвращаем True, как будто можем выполнить любой приказ.
    # В реальных сценариях нужно вернуть False, если приказ не опознан
    return True


def on_weather_forecast(bot, session, user_id, interpreted_phrase, verb_form_fields):
    """
    Обработчик запросов для прогноза погоды.
    Вызывается ядром чатбота.
    :return: текст ответа, который увидит пользователь
    """
    when_arg = bot.extract_entity(u'когда', interpreted_phrase)
    return u'Прогноз погоды на момент времени "{}" сгенерирован в функции on_weather_forecast для демонстрации'.format(when_arg)


def on_check_emails(bot, session, user_id, interpreted_phrase, verb_form_fields):
    """
    Обработчик запросов на проверку электронной почты (реплики типа "Нет ли новых писем?")
    """
    return u'Фиктивная проверка почты в функции on_check_email'


def on_alarm_clock(bot, session, user_id, interpreted_phrase, verb_form_fields):
    when_arg = bot.extract_entity(u'когда', interpreted_phrase)
    return u'Фиктивный будильник для "{}"'.format(when_arg)


def on_buy_pizza(bot, session, user_id, interpreted_phrase, verb_form_fields):
    if interpreted_phrase:
        meal_arg = bot.extract_entity(u'объект', interpreted_phrase)
        count_arg = bot.extract_entity(u'количество', interpreted_phrase)
    else:
        meal_arg = verb_form_fields['что_заказывать']
        count_arg = verb_form_fields['количество_порций']

    return u'Заказываю: что="{}", сколько="{}"'.format(meal_arg, count_arg)


def extract_last_b(dialog_path):
    if len(dialog_path) > 0 and dialog_path[-1].startswith('B:'):
        return dialog_path[-1].replace('B:', '').strip()

    return None


class Tester:
    def __init__(self, bot, user_id, output_path):
        self.bot = bot
        self.user_id = user_id
        self.wrt = io.open(output_path, 'w', encoding='utf-8')
        self.b2h = collections.defaultdict(list)
        self.hb2h = collections.defaultdict(list)
        self.brx2h = collections.defaultdict(list)
        self.default_brx2h = collections.defaultdict(list)
        self.hit_b_h = set()
        self.hit_h = set()
        self.dialog_lens = []
        self.dialogs = []
        self.default_phrase_used = False
        self.bye_phrases = []

        self.tokenizer = rutokenizer.Tokenizer()
        self.tokenizer.load()
        self.gpt_ctx_size = 2

        #self.gpt2_chitchat = None
        #self.gpt2_chitchat = Gpt2Chitchat('/home/inkoziev/github/gpt-2/models/chitchat', self.tokenizer)
        self.gpt2_chitchat = RuGpt2ChitchatConnector('http://127.0.0.1', 9098)

    def normalize_b(self, s):
        s = s.strip().lower().replace('ё', 'е')
        tx = self.tokenizer.tokenize(s)
        if tx[-1] in ['.', '!', '?']:
            tx = tx[:-1]

        s = ' '.join(tx)
        return s

    def load_dialogues(self, input_path):
        logging.debug('Loading free conversation rules from "%s"...', input_path)
        with io.open(input_path, 'r', encoding='utf-8') as rdr:
            read_state = None
            Hx0 = None
            Bx = None
            Hx = None

            for iline, line in enumerate(rdr, start=1):
                s = line.strip()
                if s.startswith('#') or len(s) == 0:
                    continue

                if s in ('HBH', 'BH', 'BH_default', 'BYE'):
                    if read_state == 'reading_BYE':
                        for h in Hx:
                            self.bye_phrases.append(h)

                    elif read_state == 'reading_BH':
                        if Bx and Hx:
                            for b, is_regex in Bx:
                                for h in Hx:
                                    if is_regex:
                                        self.brx2h[b].append(h)
                                    else:
                                        self.b2h[self.normalize_b(b)].append(h)

                    elif read_state == 'reading_BH_default':
                        if Bx and Hx:
                            for b, is_regex in Bx:
                                for h in Hx:
                                    if is_regex:
                                        self.default_brx2h[b].append(h)
                                    else:
                                        self.default_b2h[self.normalize_b(b)].append(h)

                    elif read_state == 'reading_HBH':
                        if Hx0 and Bx and Hx:
                            for h0 in Hx0:
                                for b, is_regex in Bx:
                                    assert(is_regex == 0)
                                    for h in Hx:
                                        self.hb2h[(h0, b)].append(h)

                    if s == 'HBH':
                        read_state = 'reading_HBH'
                        Hx0 = set()
                        Bx = set()
                        Hx = set()
                    elif s == 'BH':
                        read_state = 'reading_BH'
                        Bx = set()
                        Hx = set()
                    elif s == 'BH_default':
                        read_state = 'reading_BH_default'
                        Bx = set()
                        Hx = set()
                    elif s == 'BYE':
                        read_state = 'reading_BYE'
                        Bx = set()
                        Hx = set()

                elif s.startswith('B:'):
                    if read_state in ('reading_BH', 'reading_BH_default'):
                        bs = s.replace('B:', '').strip()
                        if bs.startswith('~'):
                            Bx.add((bs[1:].strip(), 1))
                        else:
                            Bx.update((z.strip(), 0) for z in bs.split('|'))
                    elif read_state == 'reading_HBH':
                        if len(Bx) != 0:
                            logging.error('Error at line #%d: len(Bx) must not be 0!', iline)
                            exit(0)

                        bs = s.replace('B:', '').strip()
                        if bs.startswith('~'):
                            raise NotImplementedError()
                        else:
                            Bx.update((z.strip(), 0) for z in bs.split('|'))
                elif s.startswith('H:'):
                    if read_state in ('reading_BH', 'reading_BH_default', 'reading_BYE'):
                        Hx.update(z.strip() for z in s.replace('H:', '').strip().split('|'))
                    elif read_state == 'reading_HBH':
                        hx = [z.strip() for z in s.replace('H:', '').strip().split('|')]
                        if not Hx0:
                            Hx0.update(hx)
                        else:
                            Hx.update(hx)
                    else:
                        raise NotImplementedError()
                else:
                    print('Unrecognized conversation rule in line {}, file="{}"'.format(iline, input_path))
                    raise NotImplementedError()

            if read_state == 'reading_BH':
                if Bx and Hx:
                    for b, is_regex in Bx:
                        for h in Hx:
                            if is_regex:
                                self.brx2h[b].append(h)
                            else:
                                self.b2h[b].append(h)
            elif read_state == 'reading_BH_default':
                if Bx and Hx:
                    for b, is_regex in Bx:
                        for h in Hx:
                            if is_regex:
                                self.default_brx2h[b].append(h)
                            else:
                                self.default_b2h[b].append(h)

            elif read_state == 'reading_HBH':
                if Hx0 and Bx and Hx:
                    for h0 in Hx0:
                        for b, is_regex in Bx:
                            assert (is_regex == 0)
                            for h in Hx:
                                self.hb2h[(h0, b)].append(h)
            elif read_state == 'reading_BYE':
                for h in Hx:
                    self.bye_phrases.append(h)

        return

    def close(self):
        self.wrt.flush()
        self.wrt.close()

    def test3(self, ruler_path):
        self.wrt.write('\n---=== BEGINNING OF DIALOGUES-3 ===----\n\n')
        with io.open(ruler_path, 'r', encoding='utf-8') as rdr:
            reader_status = ''
            for line in rdr:
                s = line.strip()
                if s.startswith('#'):
                    self.wrt.write('{}\n'.format(s))
                elif s.startswith('H:'):
                    if reader_status == '':
                        # Если боту есть что сказать по прошедшему диалогу, то вытащим его реплики
                        while True:
                            b = self.bot.pop_phrase(self.user_id)
                            if len(b) == 0:
                                break
                            self.wrt.write('B: {}\n'.format(b))

                        # Начинается новый тестовый диалог.
                        self.wrt.write('\n\n')
                        self.bot.reset_session(self.user_id)
                        self.bot.cancel_all_running_items(self.user_id)
                        self.bot.reset_usage_stat()
                        self.bot.reset_added_facts()
                    elif reader_status == 'B_processed':
                        pass
                    else:
                        raise RuntimeError()

                    h = s.replace('H:', '').strip()
                    self.bot.push_phrase(self.user_id, h)
                    self.wrt.write('H: {}\n'.format(h))
                    reader_status = 'H_processed'
                elif s.startswith('B:'):
                    reader_status = 'B_processed'
                    b = s.replace('B:', '').strip()
                    if b == '*':
                        while True:
                            b = self.bot.pop_phrase(self.user_id)
                            if len(b) == 0:
                                break
                            self.wrt.write('B: {}\n'.format(b))
                    else:
                        # TODO - тут сделать проверку, что получен указанный ответ
                        while True:
                            b = self.bot.pop_phrase(self.user_id)
                            if len(b) == 0:
                                break
                            self.wrt.write('B: {}\n'.format(b))

                elif s == '':
                    # пустая строка заканчивает диалог
                    reader_status = ''
                    # вытащим оставшиеся реплики бота
                    while True:
                        b = self.bot.pop_phrase(self.user_id)
                        if len(b) == 0:
                            break
                        self.wrt.write('B: {}\n'.format(b))
                else:
                    raise RuntimeError()

        self.wrt.write('\n---=== END OF DIALOGUES-3 ===----\n\n')
        self.wrt.flush()

    def test_free_dialogues(self, n_dial=100):
        self.wrt.write('\n\n--== FREE DIALOGUES ==--\n\n')
        for idialog in range(1, n_dial):
            self.test_starting_conversation(idialog)


        avg_len = np.mean(self.dialog_lens)
        max_len = np.max(self.dialog_lens)
        self.wrt.write('\n\n'+'#'*50)
        self.wrt.write('\n\nMetrics:\n')
        self.wrt.write('Average length={}\n'.format(avg_len))
        self.wrt.write('Max length={} for conversation:\n\n'.format(max_len))
        for d in self.dialogs:
            if len(d) == max_len:
                self.wrt.write('\n'.join(d))
                self.wrt.write('\n')
                break

        self.wrt.write('\n')

    def store_dialog(self, dialog_id, dialog_path):
        self.wrt.write('\n\n# {} dialog_id={} {}\n\n'.format('='*10, dialog_id, '='*10))

        # Распечатываем обмен репликами в этой диалоговой сессии
        self.wrt.write('{}\n'.format('\n'.join(dialog_path)))

        # 17.01.2021 дополнительно выводим всякую статистику для диалоговой сессии
        self.wrt.write('\n')
        for line in self.bot.get_session_stat(self.user_id):
            self.wrt.write('# {}\n'.format(line))
        self.wrt.write('\n\n')

        self.wrt.flush()
        self.dialogs.append(dialog_path)
        self.dialog_lens.append(len(dialog_path))

    def weight_h(self, h_text, dialog_path):
        sims = []
        for line in dialog_path[::-1][:3]:
            person, dtext = self.split_dialog_line(line)
            sim = jaccard(h_text, dtext, shingle_len=3) + 1e-5
            sims.append(sim)
        return np.mean(sims)

    def find_suitable_h_phrase2(self, prev_h, last_b, dialog_path):
        h = None
        hx = []
        hb = one_of2((prev_h, last_b), self.hb2h.keys())
        if hb:
            hx.extend(self.hb2h[hb])

        if hx:
            # Исключим H-фразы, которые уже использовались в контексте этой B-фразы
            hx = [h for h in hx if (last_b, h) not in self.hit_b_h]

            if hx:
                #h = random.choice(hx)
                eps = np.finfo(float).eps
                px = np.asarray([self.weight_h(h, dialog_path)+eps for h in hx])
                px = px / np.sqrt(np.sum(px**2))
                px /= px.sum()
                h = np.random.choice(hx, p=px)


        return h

    def find_suitable_h_phrase(self, last_b, dialog_path):
        h = None

        hx = []
        ub = self.normalize_b(last_b)
        b = one_of(ub, self.b2h.keys())
        if b:
            hx.extend(self.b2h[ub])

        if hx:
            # Исключим H-фразы, которые уже использовались в контексте этой B-фразы
            hx = [h for h in hx if ((last_b, h) not in self.hit_b_h and h not in self.hit_h)]

        # проверим по списку регулярок
        # 12-12-2020 если сработало правило на точном соответствии, то не используем регулярки
        if len(hx) == 0:
            for br in self.brx2h.keys():
                if re.search(br, last_b, re.IGNORECASE):
                    hx.extend(self.brx2h[br])

        if hx:
            # Исключим H-фразы, которые уже использовались в контексте этой B-фразы
            hx = [h for h in hx if ((last_b, h) not in self.hit_b_h and h not in self.hit_h)]

            if hx:
                # Для каждой фразы вычислим ее похожесть на последние несколько реплик в диалоге
                eps = np.finfo(float).eps
                px = np.asarray([self.weight_h(h, dialog_path)+eps for h in hx])
                px = px / np.sqrt(np.sum(px**2))
                px /= px.sum()
                h = np.random.choice(hx, p=px)

        # НАЧАЛО ОТЛАДКИ
        #if not h:
        #    if last_b in ['сколько тебе лет?', 'какое у тебя хобби?', 'чем ты увлекаешься?']:
        #        print('DEBUG@414')
        # КОНЕЦ ОТЛАДКИ

        if not h and self.default_phrase_used is False:
            hx = []
            for br in self.default_brx2h.keys():
                if re.search(br, last_b, re.IGNORECASE):
                    hx.extend(self.default_brx2h[br])

            if hx:
                h = random.choice(hx)
                self.default_phrase_used = True

        return h

    def split_dialog_line(self, line):
        i = line.index(':')
        person = line[:i]
        phrase = line[i+1:].strip()
        return person, phrase

    def extract_dialog_context(self, lines):
        cur_person, phrase = self.split_dialog_line(lines[0])
        cur_person_phrases = [phrase]
        context = []
        for line in lines[1:]:
            person, phrase = self.split_dialog_line(line)
            if person == cur_person:
                cur_person_phrases.append(phrase)
            else:
                context.append((cur_person, cur_person_phrases))
                cur_person = person
                cur_person_phrases = [phrase]

        if cur_person_phrases:
            context.append((cur_person, cur_person_phrases))

        context2 = [' . '.join(phrases) for person, phrases in context]
        return context2

    def test_starting_conversation(self, dialog_id):
        # Сбросим дискурс, чтобы вести диалог с чистого листа.
        #self.bot.reset_session(self.user_id)
        #self.bot.cancel_all_running_items(self.user_id)
        self.bot.reset_usage_stat()
        self.bot.reset_added_facts()
        self.default_phrase_used = False
        self.hit_h = set()

        interlocutor = self.user_id + str(dialog_id)

        logging.info('START DIALOG dialog_id=%d interlocutor=%s', dialog_id, interlocutor)

        # запускаем начальный диалог в боте
        self.bot.start_conversation(interlocutor)
        flush_logging()

        # Получим стартовые реплики бота (обычно он здоровается)
        dialog_path = []

        nb_gpt = 0
        said_bye = False  # флаг прощания

        do_loop = True
        while do_loop:
            do_loop = False
            last_b = None
            while True:
                b = self.bot.pop_phrase(interlocutor)
                if len(b) == 0:
                    break
                logging.debug('B: %s', b)
                dialog_path.append('B:      {}'.format(b))
                last_b = b

            if said_bye:
                # Завершен процесс прощания.
                break

            # Ищем H-фразу для полученной от бота B-реплики
            if last_b:
                h = None

                if len(dialog_path) > 1:
                    prev_h = dialog_path[-2]
                    if prev_h.startswith('H:'):
                        prev_h = prev_h.replace('H:', '').strip()
                        h = self.find_suitable_h_phrase2(prev_h, last_b, dialog_path)
                        if h:
                            logging.debug('find_suitable_h_phrase2 "%s" + "%s" ==> "%s"', prev_h, last_b, h)

                if not h:
                    h = self.find_suitable_h_phrase(last_b, dialog_path)
                    if h:
                        logging.debug('find_suitable_h_phrase "%s" ==> "%s"', last_b, h)

                if h:
                    # Найдена подходящая реплика собеседника, передаем ее боту.
                    dialog_path.append('H:      {}'.format(h))
                    self.bot.push_phrase(interlocutor, h)
                    self.hit_b_h.add((last_b, h))
                    self.hit_h.add(h)
                    do_loop = True
                else:
                    # С помощью правил не получилось сгенерировать ответную реплику.
                    logging.debug('No rule found to reply to last_b="%s"', last_b)
                    chitchat_produced = False
                    if nb_gpt < 3 and self.gpt2_chitchat is not None:
                        # не нашлось готовой реплики. используем сервис чит-чата для генерации реплики.
                        context = self.extract_dialog_context(dialog_path)
                        context = context[-self.gpt_ctx_size:]
                        logging.debug('Running gpt2 chitchat with context "%s"', ' | '.join(context))
                        hx = self.gpt2_chitchat.reply(context)
                        if hx:
                            # Выбираем вариант в текущем контексте.
                            # Для этого для каждого варианта вычислим его похожесть на последние несколько реплик в диалоге
                            eps = np.finfo(float).eps
                            px = np.asarray([self.weight_h(h, dialog_path) + eps for h in hx])
                            px = px / np.sqrt(np.sum(px ** 2))
                            px /= px.sum()
                            h = np.random.choice(hx, p=px)

                            logging.debug('gtp2 chitchat reply="%s"', h)
                            dialog_path.append('H(GPT): {}'.format(h))
                            self.bot.push_phrase(interlocutor, h)
                            self.hit_h.add(h)
                            do_loop = True
                            nb_gpt += 1
                            chitchat_produced = True
                        else:
                            logging.error('chitchat service does not generate reply')
                            do_loop = False

                        # НАЧАЛО ОТЛАДКИ
                        #print('DEBUG@464 context={} reply={}'.format(' | '.join(context), h))
                        #exit(0)
                        # КОНЕЦ ОТЛАДКИ
                    else:
                        logging.debug("Could not use chitchat service because of limit: nb_gpt=%d", nb_gpt)

                    if not chitchat_produced:
                        # Попрощаемся
                        h = random.choice(self.bye_phrases)
                        logging.debug('H(bye): last_b="%s" reply="%s"', last_b, h)
                        dialog_path.append('H(bye): {}'.format(h))
                        self.bot.push_phrase(interlocutor, h)
                        self.hit_h.add(h)
                        do_loop = True
                        said_bye = True

            else:
                # Бот ничего не выдал.

                if dialog_path and dialog_path[-1].startswith('H:'):
                    # Путь он попробует перезапустить диалог.
                    # Для этого дадим ему пустую реплику - сигнал, что в диалоге повисла "пауза"
                    self.bot.push_phrase(interlocutor, '')

                    last_b = None
                    while True:
                        b = self.bot.pop_phrase(interlocutor)
                        if len(b) == 0:
                            break
                        logging.debug('B(...): %s', b)
                        dialog_path.append('B(...): {}'.format(b))
                        last_b = b

                    if last_b:
                        h = self.find_suitable_h_phrase(last_b, dialog_path)
                        if h:
                            # Найдена подходящая реплика собеседника, передаем ее боту.
                            logging.debug('find_suitable_h_phrase "%s" ==> "%s"', last_b, h)
                            dialog_path.append('H:      {}'.format(h))
                            self.bot.push_phrase(interlocutor, h)
                            self.hit_b_h.add((last_b, h))
                            self.hit_h.add(h)
                            do_loop = True

                if last_b is None:
                    # завершаем диалог.
                    if said_bye:
                        logging.debug('Exiting conversation #%d', dialog_id)
                    else:
                        # Попрощаемся
                        h = random.choice(self.bye_phrases)
                        logging.debug('DEBUG@577 last_b is None, exiting the conversation with reply "%s"', h)
                        dialog_path.append('H(bye): {}'.format(h))
                        self.bot.push_phrase(interlocutor, h)
                        self.hit_h.add(h)
                        do_loop = True
                        said_bye = True

                    break

        logging.info('END DIALOG dialog_id=%d', dialog_id)
        self.store_dialog(dialog_id, dialog_path)
        self.bot.prune_sessions()


class ChatbotProxyObject:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def start_conversation(self, user_id):
        response = requests.get(self.endpoint_url + '/' + 'start_conversation?user={}'.format(user_id))
        if response.ok:
            return
        else:
            raise RuntimeError(response.error)

    def prune_sessions(self):
        pass

    def reset_session(self, user_id):
        pass

    def reset_usage_stat(self):
        pass

    def reset_added_facts(self):
        pass

    def get_session_stat(self, user_id):
        return []

    def cancel_all_running_items(self, user_id):
        response = requests.get(self.endpoint_url + '/' + 'cancel_all_running_items?user={}'.format(user_id))
        if response.ok:
            return
        else:
            raise RuntimeError(response.error)

    def push_phrase(self, user_id, phrase):
        response = requests.get(self.endpoint_url + '/' + 'push_phrase?user={}&phrase={}'.format(user_id, phrase))
        if response.ok:
            return
        else:
            raise RuntimeError(response.error)

    def pop_phrase(self, user_id):
        response = requests.get(self.endpoint_url + '/' + 'pop_phrase?user={}'.format(user_id))
        if response.ok:
            reply = response.json()['reply']
            return reply
        else:
            raise RuntimeError(response.error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chatbot testing')
    parser.add_argument('--scenario', type=str, choices='dialogues qa'.split())
    parser.add_argument('--bot_url', type=str, default=None)
    parser.add_argument('--user_id', type=str, default='tester')
    parser.add_argument('--data_folder', type=str, default='../../data')
    parser.add_argument('--w2v_folder', type=str, default='../../tmp')
    parser.add_argument('--profile', type=str, default='../../data/profile_1.json', help='path to profile file')
    parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--tmp_folder', type=str, default='../../tmp', help='path to folder for logfile etc')
    parser.add_argument('--input', type=str, default='../../data/test/test_phrases.txt', help='Input file with test phrases for scenario=qa')
    parser.add_argument('--output', type=str, default='../../tmp/chatbot_tester.output.txt', help='Conversations texts')

    args = parser.parse_args()
    profile_path = os.path.expanduser(args.profile)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)
    output_path = os.path.expanduser(args.output)
    user_id = args.user_id

    init_trainer_logging(os.path.join(tmp_folder, 'console_tester.log'), True)

    if args.bot_url is not None:
        logging.info('Creating proxy object for chatbot on endpoint=%s', args.bot_url)
        bot = ChatbotProxyObject(args.bot_url)
    else:
        logging.debug('In-process bot loading...')

        # Параметры работы сервиса чит-чата. По умолчанию конструктор задает разумные параметры, но для
        # удобства экспериментов повторим их тут.
        rugpt_chitchat_config = ChitchatConfig()
        #rugpt_chitchat_config.service_endpoint = 'http://127.0.0.1:9098'
        #rugpt_chitchat_config.temperature = 0.9
        rugpt_chitchat_config.num_return_sequences = 2

        bot = create_chatbot(profile_path, models_folder, w2v_folder, data_folder,
                             debugging=True,
                             chitchat_config=rugpt_chitchat_config
                             )

        # Выполняем привязку обработчиков
        bot.on_process_order = None  #on_order
        bot.add_event_handler(u'weather_forecast', on_weather_forecast)
        bot.add_event_handler(u'check_emails', on_check_emails)
        bot.add_event_handler(u'alarm_clock', on_alarm_clock)
        bot.add_event_handler(u'buy_pizza', on_buy_pizza)

    if args.scenario == 'dialogues':
        tester = Tester(bot, user_id, output_path)

        tester.load_dialogues(os.path.join(data_folder, 'test', 'generated_BH_test_rules.txt'))
        tester.load_dialogues(os.path.join(data_folder, 'test', 'test_dialogues.txt'))

        logging.debug('Starting test_free_dialogues...')
        tester.test_free_dialogues(n_dial=100)

        logging.debug('Starting test3...')
        tester.test3(os.path.join(data_folder, 'test', 'test_dialogues3.txt'))

        tester.close()
    elif args.scenario == 'qa':
        bot.start_conversation(user_id)

        # Пакетный режим qa - читаем фразы собеседника из указанного текстового файла, прогоняем
        # через бота, сохраняем ответные фразы бота в выходной файл.
        # Каждый вопрос обрабатывается изолированно, не в контексте диалога.
        with io.open(args.input, 'r', encoding='utf-8') as rdr,\
             io.open(args.output, 'w', encoding='utf-8') as wrt:

            # Получим стартовые реплики бота (обычно он здоровается)
            while True:
                answer = bot.pop_phrase(user_id)
                if len(answer) == 0:
                    break
                wrt.write('B: {}\n'.format(answer))

            # Теперь читаем фразы из тестового набора и даем их боту на обработку
            processed_phrases = set()
            for line in rdr:
                inline = line.strip()
                if inline.startswith('#'):
                    # комментарии просто сохраняем в выходном файле для
                    # удобства визуальной организации
                    wrt.write('\n{}\n'.format(inline))
                    continue

                if inline:
                    bot.cancel_all_running_items(user_id)
                    bot.reset_session(user_id)
                    bot.reset_usage_stat()
                    bot.reset_added_facts()

                    wrt.write('\nH: {}\n'.format(inline))
                    if inline not in processed_phrases:
                        processed_phrases.add(inline)
                        bot.push_phrase(user_id, inline)

                        while True:
                            answer = bot.pop_phrase(user_id)
                            if len(answer) == 0:
                                break
                            wrt.write('B: {}\n'.format(answer))

    logging.info('Tests completed.')
