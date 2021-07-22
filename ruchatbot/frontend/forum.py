"""
вободный диалог двух (в будущем - больше) экземпляров чатбота с разными профилями.
Часть проекта чатбота https://github.com/Koziev/chatbot
"""

import os
import argparse
import logging
import io
import random
import json
import numpy as np

from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.frontend.bot_creator import create_chatbot, ChitchatConfig


def reset_bot(bot, user_id):
    bot.reset_session(user_id)
    bot.cancel_all_running_items(user_id)
    bot.reset_usage_stat()
    bot.reset_all_facts()  # чтобы факты профиля перегенерировались в каждом диалоге, сбрасываем весь кэш
    #bot.reset_added_facts()


def produce_dialogue(dialog_id, bot1, user1, bot2, user2, max_turns):
    forum_logger.info('START OF DIALOG dialog_id=%d', dialog_id)

    # Сбросим дискурс, чтобы вести диалог с чистого листа.
    reset_bot(bot1, user2)  # с ботом bot1 общается собеседник user2
    reset_bot(bot2, user1)

    bots = [(bot1, user2), (bot2, user1)]

    dialog = []

    # запускаем начальный диалог в боте.
    # поочередно будем менять заводилу.
    speaking_bot_index = None
    if 0 == (dialog_id % 2):
        bot1.start_conversation(user2)
        speaking_bot_index = 0
    else:
        bot2.start_conversation(user1)
        speaking_bot_index = 1

    # Счетчик попыток принудительного продолжения диалога.
    request_continuation_counter = 0

    turn_counter = 0
    while True:
        turn_counter += 1

        speaker = bots[speaking_bot_index][0]
        speaker_name = bots[speaking_bot_index][0].get_bot_id()

        listener = bots[1-speaking_bot_index][0]
        listener_name = listener.get_bot_id()

        forum_logger.debug('--- START OF TURN #%d speaker=%s --> listener=%s ---', turn_counter, speaker_name, listener_name)

        # Извлечем все реплики из текущего бота
        replies = []
        while True:
            reply = speaker.pop_phrase(listener_name)
            if reply:
                replies.append(reply)
            else:
                break

        if len(replies) == 1:
            forum_logger.debug('Bot=%s says 1 reply="%s" to bot=%s', speaker_name, replies[0], listener_name)
        else:
            forum_logger.debug('Bot=%s says %d replies to bot=%s', speaker_name, len(replies), listener_name)
            for i, r in enumerate(replies):
                forum_logger.debug('reply[%d]: %s', i, r)

        if len(replies) == 0:
            # Диалог остановился, так как speaker не выдал никакой реплики.

            f = ""
            dialog.append((speaker_name, f))

            # Слушатель прощается
            f = random.choice(bye_phrases)
            dialog.append((listener_name, f))
            speaker.push_phrase(user_id=listener_name, question=f)
            break
        elif len(dialog) >= max_turns:
            # Попрощаемся от имени второго бота (который сейчас слушатель), если если диалог слишком длинный.
            f = random.choice(bye_phrases)
            dialog.append((listener_name, f))
            speaker.push_phrase(user_id=listener_name, question=f)
            break
        else:
            for r in replies:
                dialog.append((speaker_name, r))

            # теперь все реплики склеиваем в одно длинное высказывание и подаем на вход другого бота
            merged_reply = ''
            for reply in replies:
                if merged_reply:
                    if merged_reply[-1] not in '.?!':
                        merged_reply += '.'
                    merged_reply += ' '

                merged_reply += reply

            forum_logger.debug('Bot %s sends "%s" to bot %s', speaker_name, merged_reply, listener_name)
            listener.push_phrase(speaker_name, merged_reply)

        forum_logger.debug('--- END OF TURN #%d ---', turn_counter)

        # переключаемся на другой бот
        speaking_bot_index = 1 - speaking_bot_index

    forum_logger.info('END OF DIALOG dialog_id=%d', dialog_id)

    return dialog


def extract_interlocutor_name(profile_path):
    # Для удобства чтения логов попробуем выделить имя аватара из секции констант профиля бота.
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        if 'name_nomn' in profile['constants']:
            return profile['constants']['name_nomn']

    # Дефолтное обозначение собеседника получаем как имя файла профиля без расширения
    return os.path.basename(profile_path).replace('.json', '')


def calc_discovery_metric(bot1, user1, bot2, user2):
    """ Расчет DISCOVERY метрики, показывающей, насколько эффективно бот №1 узнал факты из профиля бота №2 """

    text_utils = bot1.get_engine().get_text_utils()

    session1 = bot1.get_session(user2)
    discovered_facts1 = []

    for fact0, section, src in session1.facts_storage.get_added_facts(user2):
        if src == '--from dialogue--':
            fact = bot1.get_engine().interpreter.flip_person(fact0, text_utils)
            fact = text_utils.wordize_text(fact)
            discovered_facts1.append((fact, section, src))

    session2 = bot2.get_session(user1)
    facts2 = session2.facts_storage
    hidden_facts = [(text_utils.wordize_text(z[0]), z[1], z[2]) for z in facts2.enumerate_facts(user1) if z[1] == '1s']

    synonymy_detector = bot1.get_engine().synonymy_detector

    # Маппинг извлеченных фактов на скрытые факты у собеседника
    fact_sims = []
    for f, _, _ in discovered_facts1:
        best_hidden_fact, best_sim = synonymy_detector.get_most_similar(f, hidden_facts, text_utils, nb_results=1)
        fact_sims.append(best_sim)

    precision = np.mean(fact_sims)+1e-10

    # Маппинг скрытых фактов у собеседника на извлеченные факты
    fact_sims = []
    for f, _, _ in hidden_facts:
        best_hidden_fact, best_sim = synonymy_detector.get_most_similar(f, discovered_facts1, text_utils, nb_results=1)
        fact_sims.append(best_sim)

    recall = np.mean(fact_sims)+1e-10

    f1 = 2 / (1/recall + 1/precision)

    return precision, recall, f1


class BotMetrics:
    def __init__(self):
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def store(self, precision, recall, f1):
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1s.append(f1)

    def precision(self):
        return np.mean(self.precisions)

    def recall(self):
        return np.mean(self.recalls)

    def f1(self):
        return np.mean(self.f1s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chatbots conversations')
    parser.add_argument('--data_folder', type=str, default='../../../data')
    parser.add_argument('--w2v_folder', type=str, default='../../../tmp')
    parser.add_argument('--profile1', type=str, default='../../../data/profile_1.json')
    parser.add_argument('--profile2', type=str, default='../../../data/profile_2.json')
    parser.add_argument('--models_folder', type=str, default='../../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--tmp_folder', type=str, default='../../../tmp', help='path to folder for logfile etc')
    parser.add_argument('--test_dialogues', type=str, default='../../../data/test/test_dialogues.txt')
    parser.add_argument('--output', type=str, default='../../../tmp/forum.output.txt', help='Conversation scripts')

    args = parser.parse_args()
    profile1_path = os.path.expanduser(args.profile1)
    profile2_path = os.path.expanduser(args.profile2)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)
    output_path = os.path.expanduser(args.output)

    init_trainer_logging(os.path.join(tmp_folder, 'forum.log'), True)
    forum_logger = logging.getLogger('Forum')

    user1 = extract_interlocutor_name(profile1_path)
    user2 = extract_interlocutor_name(profile2_path)

    nb_dialogues = 100  # столько диалогов будет создано

    forum_logger.info('interlocutor#1: profile=%s name=%s', profile1_path, user1)
    forum_logger.info('interlocutor#2: profile=%s name=%s', profile2_path, user2)

    bye_phrases = []
    with io.open(args.test_dialogues, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s == 'BYE':
                for line in rdr:
                    s = line.strip()
                    if s:
                        f = s.replace('H:', '').strip()
                        bye_phrases.append(f)
                    else:
                        break
                break


    # Параметры работы сервиса чит-чата. По умолчанию конструктор задает разумные параметры, но для
    # удобства экспериментов повторим их тут.
    rugpt_chitchat_config = ChitchatConfig()
    #rugpt_chitchat_config.service_endpoint = 'http://127.0.0.1:9098'
    #rugpt_chitchat_config.temperature = 0.9
    rugpt_chitchat_config.num_return_sequences = 2

    forum_logger.debug('Create bot #1 using %s...', profile1_path)
    bot1 = create_chatbot(profile1_path, models_folder, w2v_folder, data_folder,
                          bot_id=user1,
                          debugging=False,
                          chitchat_config=rugpt_chitchat_config)

    forum_logger.debug('Create bot #2 using %s...', profile2_path)
    bot2 = create_chatbot(profile2_path, models_folder, w2v_folder, data_folder,
                          bot_id=user2,
                          debugging=False,
                          chitchat_config=rugpt_chitchat_config)

    usr_len = max(len(user1), len(user2))

    bot1_metrics = BotMetrics()
    bot2_metrics = BotMetrics()

    with io.open(output_path, 'w', encoding='utf-8') as wrt:
        for dialog_id in range(1, nb_dialogues+1):
            # Начинается новый диалог.
            dialog = produce_dialogue(dialog_id, bot1, user1, bot2, user2, max_turns=50)

            if dialog_id > 1:
                wrt.write('\n'*4)
            wrt.write('# =============== dialog_id={} ===============\n\n'.format(dialog_id))
            for speaker, phrase in dialog:
                wrt.write('{} :> {}\n'.format(speaker.ljust(usr_len), phrase))

            # 19.01.2021 дополнительно выводим всякую статистику для обеих диалоговых сессий
            wrt.write('\n')
            wrt.write('# Statistics for bot "{}" session:\n'.format(bot1.get_bot_id()))
            for line in bot1.get_session_stat(user2):
                wrt.write('# {}\n'.format(line))

            precision, recall, f1 = calc_discovery_metric(bot1, user1, bot2, user2)
            bot1_metrics.store(precision, recall, f1)
            wrt.write('Discovery metrics: precision={} recall={} f1={}\n'.format(precision, recall, f1))

            wrt.write('\n')
            wrt.write('# Statistics for bot "{}" session:\n'.format(bot2.get_bot_id()))
            for line in bot2.get_session_stat(user1):
                wrt.write('# {}\n'.format(line))

            precision, recall, f1 = calc_discovery_metric(bot2, user2, bot1, user1)
            bot2_metrics.store(precision, recall, f1)
            wrt.write('Discovery metrics: precision={} recall={} f1={}\n'.format(precision, recall, f1))

            wrt.write('\n'*4)
            wrt.flush()

    forum_logger.info('%d conversions finished.', nb_dialogues)
    forum_logger.info('mean discovery metrics for bot "%s": precision=%f recall=%f f1=%f',
                      bot1.get_bot_id(), bot1_metrics.precision(), bot1_metrics.recall(), bot1_metrics.f1())

    forum_logger.info('mean discovery metrics for bot "%s": precision=%f recall=%f f1=%f',
                      bot2.get_bot_id(), bot2_metrics.precision(), bot2_metrics.recall(), bot2_metrics.f1())




