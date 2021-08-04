"""
Модель для генерации реплик в чит-чате
Часть пайплайна чатбота https://github.com/Koziev/chatbot
"""

import logging.handlers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DialoGPT3Ru:
    def __init__(self):
        print('#rugpt_chitchat.py -> init')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger('RugptChitChat')
        self.temperature = 0.9
        self.beam_k = 10
        self.beam_p = 0.9
        self.repetition_penalty = 1.0

    def load(self, model_name_or_path):
        print('#rugpt_chitchat.py -> load:', model_name_or_path)
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    ## Added
    def get_length_param(self, text: str) -> str:
        model_tokenizer = self.tokenizer
        tokens_count = len(model_tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    def generate_output(self, context, num_return_sequences=10):
        print('#rugpt_chitchat.py -> generate_output:', num_return_sequences, context)
        self.logger.debug('Generating DialoGPT3Ru response with context=%s', context)

        nspaces = context.count(' ') # TODO AK наверное лучше считать по токенам
        if nspaces == 0:
            # По однословным контекстам не будем генерировать отклик.
            print(f'#rugpt3_DialoGPT3Ru.py -> generate_output: однословный контекст {nspaces}. пропускаем GPT')
            return []

        # prompt_text = context + ' |'
        stop_token = "</s>"
        length = 50 #80

        # История генерируется в контексте
        # if len(full_inputs) > 5: # Ограничиваем кол-во истории в диалоге на 2 предыдущих + вопрос
        #   inputs = full_inputs[-5:]
        # else:
        #   inputs = full_inputs
        # inputs_text = ''
        # for input_ in inputs:
        #   if params['is_always_use_length']:
        #       length_param = get_length_param(input_['text'], dialog_tokenizer)
        #   else:
        #       length_param = '-'
        #   inputs_text += f"|{input_['speaker']}|{length_param}|{input_['text']}"
        # inputs_text += f"|1|{params['length_generate']}|"

        if '|' in context:
            phrases = context.split('|')
            isStartHuman = True if len(phrases) % 2 != 0 else False
            prompt_text = ''
            for idx, text in enumerate(phrases):
                print('#', idx, text)
                length_param = self.get_length_param(text)
                person_id = 0 if (isStartHuman and idx % 2 == 0) or (not isStartHuman and idx % 2 != 0) else 1
                prompt_text += f'|{person_id}|{length_param}|{text}'
            prompt_text += f'|1|2|'
        else:
            length_param = self.get_length_param(context)
            prompt_text = f'|0|{length_param}|{context}|1|2|' # 2 ожидаем ответ до 50
        print('# prompt_text', prompt_text)

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)
        print('# len encoded_prompt',len(encoded_prompt))

        # output_sequences = self.model.generate(
        #     input_ids=encoded_prompt,
        #     max_length=length, # + len(encoded_prompt[0]),
        #     temperature=self.temperature,
        #     top_k=self.beam_k,
        #     top_p=self.beam_p,
        #     repetition_penalty=self.repetition_penalty,
        #     do_sample=True,
        #     num_return_sequences=num_return_sequences,
        #     # pad_token_id=50256,  # ой нехорошо, но ворнинг достал
        # )
        # print('#rugpt_chitchat.py -> generate_output -> output_sequences:', output_sequences)
        try:
            # ToDo make this asynchronous
            outputs_token_ids = self.model.generate(
                input_ids=encoded_prompt,
                # inputs_token_ids,
                max_length=length + len(encoded_prompt[0]),
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.95,
                temperature=0.6,
                num_return_sequences=num_return_sequences,
                device='cpu', # "cuda" if torch.cuda.is_available() else "cpu",
                mask_token_id=self.tokenizer.mask_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                unk_token_id=self.tokenizer.unk_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        except Exception as e:
            # print(f"===> Error generate: {str(e)}")
            print(f"===> Error generate: ", e)
            # return {'inputs': '', 'outputs': '', 'status': False, 'msg': f"{str(e)}"}

        # TODO проверить. Если включать skip_special_tokens, то вместо
        #  |0|1|меня зовут вася|1|2|Вася, как к Вам можно обращаться? Я могу на ты или на вы? |0-|Здравствуйте! Мне 29 лет. Есть сын 7 лет, 4 месяца назад я узнала, что беременна. Я стала спрашивать у мужа
        # Дает только вторую часть Здравствуйте! Мне 29 лет. Есть сын 7 лет, 4 месяца назад я узнала, что беременна. Я стала спрашивать у мужа, что
        # Хотя первая лучше(!) и правильно отвечат на первый вопрос,а не продложает вторую фразу(!)
        # outputs = [self.tokenizer.decode(x, skip_special_tokens=True) for x in outputs_token_ids]
        # outputs = [x.split('|')[-1] for x in outputs]
        # print('#rugpt_chitchat.py -> generate_output -> output_sequences: ПРИМЕРЫ', outputs)

        output_sequences = outputs_token_ids
        # Remove the batch dimension when returning multiple sequences
        try:
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()
        except Exception as e:
            print(f"===> RUGPT Error generate: TODO(!) взять из GPT3ru обработку", e)

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("ruGPT2Large:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

            if '#' in total_sequence:
                total_sequence = total_sequence[: total_sequence.find('#')]

            if '|' in total_sequence: # TODO удалить, после изменения логики формирования структуры. Т.к. сейчас несовместимы
                # total_sequence = total_sequence.split('|')[-1]
                total_sequence = total_sequence[: total_sequence.find('|')]
            total_sequence = total_sequence.strip()
            if '|' not in total_sequence:
                generated_sequences.add(total_sequence)
            print(generated_sequence_idx, "RUGPT total_sequence:", total_sequence)

        print('#rugpt_DialoGPT3Ru.py -> generate_output -> generated_sequences:', generated_sequences)
        self.logger.debug('DialoGPT3Ru generated %d responses: %s', len(generated_sequences), '; '.join(generated_sequences))
        return list(generated_sequences)

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    chitchat = DialoGPT3Ru()
    chitchat.load('/home/inkoziev/polygon/chatbot/tmp/rugpt_checkpoints')

    context = []
    while True:
        q = input(':> ').strip()
        if q:
            context.append(q)
        else:
            if context:
                context = ' | '.join(context)
                px = chitchat.generate_output(context)
                for p in px:
                    print('{}'.format(p))
                print('')
                context = []
