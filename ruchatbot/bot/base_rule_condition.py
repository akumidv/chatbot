# -*- coding: utf-8 -*-
"""
23-01-2021 Добавляем проверку модальности предложения в кондикторе "text": assertion | question | imperative
"""


import logging
import re

from ruchatbot.bot.keyword_matcher import KeywordMatcher
from ruchatbot.utils.constant_replacer import replace_constant
from ruchatbot.bot.phrase_token import PhraseToken
from ruchatbot.utils.chunk_tools import normalize_chunk
from ruchatbot.bot.rule_condition_matching import RuleConditionMatching


class BaseRuleCondition(object):
    def __init__(self, data_yaml):
        self.key = ';'.join(k+'|'+str(v) for (k, v) in data_yaml.items())
        self.modality = None

    def get_short_repr(self):
        return None

    def __repr__(self):
        return self.get_short_repr()

    @staticmethod
    def from_yaml(condition_yaml, constants, text_utils):
        if u'intent' in condition_yaml:
            return RuleCondition_Intent(condition_yaml)
        elif u'state' in condition_yaml:
            return RuleCondition_State(condition_yaml)
        elif 'text' in condition_yaml or 'assertion' in condition_yaml or 'question' in condition_yaml:
            return RuleCondition_Text(condition_yaml, constants, text_utils)
        elif u'raw_text' in condition_yaml:
            return RuleCondition_RawText(condition_yaml, constants, text_utils)
        elif u'prev_bot_text' in condition_yaml:
            return RuleCondition_PrevBotText(condition_yaml, constants, text_utils)
        elif u'keyword' in condition_yaml:
            return RuleCondition_Keyword(condition_yaml, constants, text_utils)
        elif u'regex' in condition_yaml:
            return RuleCondition_Regex(condition_yaml, constants, text_utils)
        elif u'match' in condition_yaml or u'match_raw' in condition_yaml:
            return RuleCondition_ChunkMatcher(condition_yaml, constants, text_utils)
        else:
            logging.error('Unknown condition keyword "%s"', list(condition_yaml.keys())[0])
            raise NotImplementedError()

        return res

    def get_key(self):
        return self.key

    def check_text(self, input_text, etalon_texts, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if not input_text:
            return False

        if self.modality:
            if self.modality == 'assertion':
                # Проверяем, что входной текст реализует утверждение, а не вопрос.
                if input_text[-1] == '?' or not interpreted_phrase.is_assertion:
                    return False
            elif self.modality == 'question':
                # Проверяем, что входной текст реализует утверждение, а не вопрос.
                if input_text[-1] in '.!' or not interpreted_phrase.is_question:
                    return False
            else:
                raise NotImplementedError()

        text_utils = answering_engine.get_text_utils()
        word_embeddings = answering_engine.get_word_embeddings()

        input_text = text_utils.wordize_text(input_text)
        etalons = list((text_utils.wordize_text(etalon), None, None) for etalon in etalon_texts)

        if self.metric == 'jaccard':
            syn = answering_engine.get_jaccard_detector()
            best_etalon, best_sim = syn.get_most_similar(input_text,
                                                         etalons,
                                                         text_utils,
                                                         nb_results=1)
        else:
            syn = answering_engine.get_synonymy_detector()
            best_etalon, best_sim = syn.get_most_similar(input_text,
                                                         etalons,
                                                         text_utils,
                                                         nb_results=1)

        # НАЧАЛО ОТЛАДКИ
        if best_sim >= syn.get_threshold():
            logging.debug('Text matched in rule: input_text="%s", best_etalon="%s", best_sim=%g', input_text, best_etalon, best_sim)
        # КОНЕЦ ОТЛАДКИ

        return best_sim >= syn.get_threshold()

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        raise NotImplementedError()


class RuleCondition_Intent(BaseRuleCondition):
    def __init__(self, data_yaml):
        super().__init__(data_yaml)
        self.ored_intents = [x.strip() for x in data_yaml[u'intent'].split('|')]

    def get_short_repr(self):
        return 'intent={}'.format('|'.join(self.ored_intents))

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if interpreted_phrase.intents is None:
            logging.error('interpreted_phrase.intents is None in RuleCondition_Intent.check_condition')
            return RuleConditionMatching.create(False)

        assert(self.ored_intents is not None)
        return RuleConditionMatching.create(any((x in self.ored_intents) for x in interpreted_phrase.intents))


class RuleCondition_State(BaseRuleCondition):
    def __init__(self, data_yaml):
        super().__init__(data_yaml)
        self.slot_name = data_yaml['state']['slot']
        self.slot_mask = data_yaml['state']['regex_mask']

    def get_short_repr(self):
        return 'state {}={}'.format(self.slot_name, self.slot_mask)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        slot_value = session.get_slot(self.slot_name)
        f = slot_value and re.match(self.slot_mask, slot_value)
        return RuleConditionMatching.create(f)


def ngrams(s, n):
    #return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))


class RuleCondition_Text(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        self.metric = 'synonymy'
        self.threshold = None
        self.modality = None

        if 'text' in data_yaml:
            y = data_yaml['text']
        elif 'assertion' in data_yaml:
            y = data_yaml['assertion']
            self.modality = 'assertion'
        elif 'question' in data_yaml:
            y = data_yaml['question']
            self.modality = 'question'
        else:
            raise NotImplementedError()

        if isinstance(y, dict):
            etalons = y['masks']
            self.metric = y.get('metric', 'synonymy')
            self.threshold = y.get('threshold')
        elif isinstance(y, list):
            etalons = y
        else:
            etalons = [y]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

    def get_short_repr(self):
        return 'text etalons[1/{}]="{}" metric={}'.format(len(self.etalons), self.etalons[0], self.metric)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        input_text = interpreted_phrase.interpretation
        f = False
        if self.metric.startswith('jaccard'):
            shingle_len = 3
            if self.metric == 'jaccard2':
                shingle_len = 2

            threshold = answering_engine.get_synonymy_detector().get_threshold()
            for etalon in self.etalons:
                if jaccard(etalon.lower().replace('ё', 'е'), input_text.lower().replace('ё', 'е'), shingle_len) >= threshold:
                    f = True
                    break
        elif self.metric == 'synonymy':
            f = self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
        else:
            raise NotImplementedError()

        return RuleConditionMatching.create(f)


class RuleCondition_RawText(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        self.metric = 'synonymy'
        self.threshold = None
        self.modality = None

        #if isinstance(data_yaml[u'raw_text'], list):
        #    etalons = data_yaml[u'raw_text']
        #else:
        #    etalons = [data_yaml[u'raw_text']]
        y = data_yaml['raw_text']
        if isinstance(y, dict):
            etalons = y['masks']
            self.metric = y.get('metric', 'synonymy')
            self.threshold = y.get('threshold')
        elif isinstance(y, list):
            etalons = y
        else:
            etalons = [y]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

    def get_short_repr(self):
        return 'raw_text etalons[1/{}]="{}"'.format(len(self.etalons), self.etalons[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        input_text = interpreted_phrase.raw_phrase
        f = self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
        return RuleConditionMatching.create(f)


class RuleCondition_PrevBotText(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)

        # Проверяем, что последняя фраза бота была похожа на указанный эталон
        if isinstance(data_yaml[u'prev_bot_text'], list):
            etalons = data_yaml[u'prev_bot_text']
        else:
            etalons = [data_yaml[u'prev_bot_text']]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

        self.metric = 'synonymy'

    def get_short_repr(self):
        return 'prev_bot_text etalons[1/{}]="{}"'.format(len(self.etalons), self.etalons[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        b = session.get_last_bot_utterance()
        if b:
            input_text = b.interpretation
            f = self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
            return RuleConditionMatching.create(f)
        else:
            return RuleConditionMatching.create(False)


class RuleCondition_Keyword(BaseRuleCondition):
    """Проверяется наличие в тексте заданных ключевых слов"""
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)

        if isinstance(data_yaml[u'keyword'], list):
            etalons = data_yaml[u'keyword']
        else:
            etalons = [data_yaml[u'keyword']]

        self.matchers = [KeywordMatcher.from_string(replace_constant(etalon, constants, text_utils))
                                                    for etalon
                                                    in etalons]

    def get_short_repr(self):
        return 'keyword matchers={}'.format(' '.join(map(str, self.matchers)))

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        f = True
        for matcher in self.matchers:
            if not matcher.match(interpreted_phrase):
                f = False
                break

        return RuleConditionMatching.create(f)


class RuleCondition_Regex(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        self.rx_str = replace_constant(data_yaml['regex'], constants, text_utils)

    def get_short_repr(self):
        return 'regex="{}"'.format(self.rx_str)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        f = re.search(self.rx_str, interpreted_phrase.interpretation, flags=re.IGNORECASE)
        return RuleConditionMatching.create(f)


class MaskTerm:
    def __init__(self):
        self.word = None
        self.norm_word = None
        self.lemma = None
        self.chunk_type = None  # NP | VI | VP | AP
        self.chunk_name = None

    def __repr__(self):
        if self.word:
            return self.word
        else:
            return '[{} {}]'.format(self.chunk_type, self.chunk_name)

    def is_chunk(self):
        return self.chunk_name is not None

    def is_NP(self):
        return self.chunk_type == 'NP'

    def is_VI(self):
        return self.chunk_type == 'VI'

    def is_VP(self):
        return self.chunk_type == 'VP'

    def is_AP(self):
        return self.chunk_type == 'AP'


class MatchResult:
    def __init__(self):
        self.total_penalty = 0
        self.term_matchings = []
        self.index2group = dict()

    def group(self, index):
        return self.index2group[index]

    def groups_count(self):
        return len(self.index2group)


class TermMatch:
    def __init__(self, term):
        self.term = term
        self.penalty = 0
        self.first_token_index = None
        self.last_token_index = None
        self.matched_token = None
        self.chunk_tokens = []

    def __repr__(self):
        s = str(self.term)
        if self.matched_token:
            s += '={}'.format(self.matched_token.word)
        else:
            s += '={}'.format(' '.join(t.word for t in self.chunk_tokens))

        if self.penalty != 0:
            s += ' (-{})'.format(self.penalty)

        return s

    def tokens_offset(self):
        return self.last_token_index - self.first_token_index + 1


class MatchChainItem:
    def __init__(self, prev_chain_item, term_match):
        self.prev_chain_item = prev_chain_item
        self.term_match = term_match

    def __repr__(self):
        return str(self.term_match) if self.term_match else '<<<NULL>>>'

    @staticmethod
    def start():
        return MatchChainItem(None, None)

    def create_match(self):
        items = []
        cur_item = self
        while cur_item:
            items.append(cur_item)
            cur_item = cur_item.prev_chain_item

        items = list(reversed(items))[1:]  # начальный элемент цепочки пропустим.

        mr = MatchResult()
        mr.total_penalty = sum(x.term_match.penalty for x in items)
        mr.term_matchings = list(x.term_match for x in items)
        for item in items:
            if item.term_match.chunk_tokens:
                mr.index2group[item.term_match.term.chunk_name] = item.term_match.chunk_tokens

        return mr


def match(phrase_tokens, mask_terms):
    chain_start = MatchChainItem.start()
    chains = all_matches(0, phrase_tokens, mask_terms, chain_start)
    if chains:
        chains = [x.create_match() for x in chains]
        chains = sorted(chains, key=lambda z: z.total_penalty)
        best_chain = chains[0]
        return best_chain
    else:
        return None


def match_term(start_token_index, phrase_tokens, mask_term):
    """ Ищем варианты сопоставления терма mask_term на цепочку токенов phrase_tokens """
    results = []
    if mask_term.is_chunk():
        # Ищем сопоставление для чанка.
        for skip in range(0, len(phrase_tokens) - start_token_index):
            phrase_token = phrase_tokens[start_token_index + skip]

            if mask_term.is_NP():
                if not phrase_token.is_pron():
                    # Сначала надо найти токен, соответствующий началу чанка.
                    if phrase_token.is_chunk_starter:
                        m = TermMatch(mask_term)
                        m.penalty = skip
                        m.chunk_tokens.append(phrase_tokens[start_token_index + skip])

                        chunk_index = phrase_tokens[start_token_index + skip].chunk_index
                        for t in phrase_tokens[start_token_index + skip + 1:]:
                            if t.chunk_index == chunk_index:
                                m.chunk_tokens.append(t)
                            else:
                                break

                        m.first_token_index = start_token_index + skip
                        m.last_token_index = start_token_index + skip + len(m.chunk_tokens) - 1

                        results.append(m)
                elif phrase_token.is_noun() or phrase_token.is_adj():
                    # Одиночное существительное может рассматриваться как часть NP-чанка-контейнера:
                    # "Восход Солнца"
                    #         ^^^^^^
                    # Если существительное не является началом именной группы, но входит в нее,
                    # то считаем все токены, начиная с этого сущ и до конца NP-чанка, сматченным фрагментом.
                    m = TermMatch(mask_term)
                    m.penalty = skip + 0.1
                    m.chunk_tokens.append(phrase_tokens[start_token_index + skip])

                    chunk_index = phrase_tokens[start_token_index + skip].chunk_index
                    for t in phrase_tokens[start_token_index + skip + 1:]:
                        if t.chunk_index == chunk_index:
                            m.chunk_tokens.append(t)
                        else:
                            break

                    m.first_token_index = start_token_index + skip
                    m.last_token_index = start_token_index + skip + len(m.chunk_tokens) - 1

                    results.append(m)
            elif mask_term.is_VI():
                if phrase_token.is_inf():
                    m = TermMatch(mask_term)
                    m.penalty = skip
                    m.chunk_tokens.append(phrase_tokens[start_token_index + skip])
                    m.first_token_index = start_token_index + skip
                    m.last_token_index = start_token_index + skip
                    results.append(m)
            elif mask_term.is_VP():
                # TODO: сейчас матчим один глагол, но надо учитывать аналитические и модальные конструкции
                if phrase_token.is_verb():
                    m = TermMatch(mask_term)
                    m.penalty = skip
                    m.chunk_tokens.append(phrase_tokens[start_token_index + skip])
                    m.first_token_index = start_token_index + skip
                    m.last_token_index = start_token_index + skip
                    results.append(m)
            elif mask_term.is_AP():
                if phrase_token.is_adj():
                    m = TermMatch(mask_term)
                    m.penalty = skip
                    m.chunk_tokens.append(phrase_tokens[start_token_index + skip])
                    m.first_token_index = start_token_index + skip
                    m.last_token_index = start_token_index + skip
                    results.append(m)
            else:
                raise NotImplementedError()

    else:
        # Ищем сопоставление для слова из маски
        for skip, token in enumerate(phrase_tokens[start_token_index:]):
            if token.lemma == mask_term.lemma:
                m = TermMatch(mask_term)
                m.matched_token = token
                m.penalty = skip
                m.first_token_index = start_token_index + skip
                m.last_token_index = start_token_index + skip
                results.append(m)

    return results


def all_matches(start_token_index, phrase_tokens, mask_terms, prev_match):
    term_matchings = match_term(start_token_index, phrase_tokens, mask_terms[0])

    res = []
    for term_matching in term_matchings:
        chain_item = MatchChainItem(prev_match, term_matching)
        if len(mask_terms) > 1:
            next_chains = all_matches(term_matching.last_token_index+1,
                                      phrase_tokens,
                                      mask_terms[1:], chain_item)
            res.extend(next_chains)
        else:
            # последний терм в шаблоне.
            # посмотрим, сколько справа осталось несопоставленных токенов во входном предложении.
            tail_len = len(phrase_tokens) - start_token_index
            #chain_item.penalty += tail_len

            res.append(chain_item)

    return res


class ChunkMatcherMask:
    def __init__(self, mask_str, constants, text_utils):
        self.mask_terms = []

        mask_str = replace_constant(mask_str, constants, text_utils)
        mask_tokens = text_utils.tokenizer.tokenize(mask_str)
        mask_tags = list(text_utils.postagger.tag(mask_tokens))
        mask_lemmas = text_utils.lemmatizer.lemmatize(mask_tags)

        for token, tags, lemma in zip(mask_tokens, mask_tags, mask_lemmas):
            term = MaskTerm()
            term.word = token
            term.norm_word = token.lower()
            term.lemma = lemma[2]
            if token.startswith('np'):
                term.chunk_type = 'NP'
                term.chunk_name = token
            elif token.startswith('vi'):
                term.chunk_type = 'VI'
                term.chunk_name = token
            elif token.startswith('vp'):
                term.chunk_type = 'VP'
                term.chunk_name = token
            elif token.startswith('ap'):
                term.chunk_type = 'AP'
                term.chunk_name = token

            self.mask_terms.append(term)

    def __repr__(self):
        return ' '.join(str(term) for term in self.mask_terms)

    def get_vars(self):
        return list(term.chunk_name for term in self.mask_terms if term.chunk_name)


class ChunkMatcher_W2V_Constraint:
    def __init__(self, anchor, sim):
        self.anchor = anchor
        self.sim = sim


class RuleCondition_ChunkMatcher(BaseRuleCondition):
    def __init__(self, mask_yaml, constants, text_utils):
        self.masks = []
        self.constraints_w2v = dict()

        if 'match' in mask_yaml:
            mask_data = mask_yaml['match']
            self.is_raw = False
        else:
            mask_data = mask_yaml['match_raw']
            self.is_raw = True

        if isinstance(mask_data, str):
            self.masks.append(ChunkMatcherMask(mask_data, constants, text_utils))
        elif isinstance(mask_data, dict):
            for mask_str in mask_data['masks']:
                self.masks.append(ChunkMatcherMask(mask_str, constants, text_utils))

            if 'constraints' in mask_data:
                for constraint1 in mask_data['constraints']:
                    # Предполагаем, что в constraint1 заданы ограничения на одну из сопоставляемых переменных.
                    var_name = list(constraint1.keys())[0]
                    constraints_str = constraint1[var_name]
                    # сейчас хардкодим поддержку только w2v-предиката, но потом надо будет сделать более
                    # универсальное решение.
                    for m_w2v in re.finditer(r'w2v\(\s*(.+)\s*,\s*([0-9.]+)\s*\)', constraints_str):
                        anchor = m_w2v.group(1).strip().lower()
                        sim = float(m_w2v.group(2).strip())
                        c = ChunkMatcher_W2V_Constraint(anchor, sim)
                        self.constraints_w2v[var_name.lower()] = [c]
        else:
            for mask_str in mask_data:
                self.masks.append(ChunkMatcherMask(mask_str, constants, text_utils))

    def __repr__(self):
        return str(self.masks[0])

    def get_short_repr(self):
        return str(self.masks[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if self.is_raw:
            input_phrase = interpreted_phrase.raw_phrase
        else:
            input_phrase = interpreted_phrase.interpretation

        text_utils = answering_engine.text_utils
        tokens = text_utils.tokenizer.tokenize(input_phrase)
        tagsets = list(text_utils.postagger.tag(tokens))
        lemmas = text_utils.lemmatizer.lemmatize(tagsets)

        #edges = syntan.parse(tokens, tagsets)
        # заглушка вместо парсинга:
        edges = [(word, iword, None, None, None) for (iword, word) in enumerate(tokens)]

        phrase_tokens = []
        for word_index, (token, tagset, lemma) in enumerate(zip(tokens, tagsets, lemmas)):
            t = PhraseToken()
            t.word = token
            t.norm_word = token.lower()
            t.lemma = lemma[2]
            t.tagset = tagset[1]
            t.word_index = word_index
            phrase_tokens.append(t)

        chunks = text_utils.chunker.parse(tokens)
        for chunk_index, chunk in enumerate(chunks):
            phrase_tokens[chunk.tokens[0].index].is_chunk_starter = True
            for token in chunk.tokens:
                phrase_tokens[token.index].chunk_index = chunk_index

        for mask in self.masks:
            mx = match(phrase_tokens, mask.mask_terms)
            if mx:
                #print('{} groups in matching:'.format(mx.groups_count()))
                res = RuleConditionMatching.create(True)
                for group_name, tokens in mx.index2group.items():
                    normal_words1 = normalize_chunk(tokens, edges, text_utils.flexer, text_utils.word2tags)
                    #normal_words2 = normalize_chunk(tokens, edges, text_utils.flexer, text_utils.word2tags, target_tags={'ЧИСЛО': 'ЕД'})
                    #normal_words = list(set(normal_words1) | set(normal_words2))
                    normal_words = normal_words1

                    if group_name in self.constraints_w2v:
                        constraints_satisfied = True
                        for c in self.constraints_w2v[group_name]:
                            hit = False
                            for chunk_word in normal_words:
                                sim = text_utils.word_similarity(c.anchor, chunk_word)
                                if sim >= c.sim:
                                    hit = True
                                    break
                            if not hit:
                                constraints_satisfied = False
                                break

                        if not constraints_satisfied:
                            return RuleConditionMatching.create(False)

                    #print('{}={} normal={}'.format(group_name, ' '.join(t.word for t in tokens), ' '.join(normal_words)))
                    res.add_group(group_name.upper(), normal_words, tokens)

                return res

        return RuleConditionMatching.create(False)
