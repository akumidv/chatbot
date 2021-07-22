import logging
import json
import requests

from flask import Flask, request, Response
from flask import jsonify


flask_app = Flask(__name__)

@flask_app.route('/webhook', methods=['POST'])
def respond():
    print(request.json)
    return Response(status=200)


chatbot_url = 'http://127.0.0.1:9001'


@flask_app.route('/ping', methods=['GET'])
def home():
    return "Webhook v.4"


@flask_app.route('/', methods=['POST', 'GET'])
def root_method():
    print('method={}'.format(request.method))

    if request.method == 'GET':
        return Response(status=200)
    elif request.method == 'POST':
        #print(request.json)
        print('request.data=', request.data)

        jdata = json.loads(request.data)

        # Во входящей структуре должны быть поля:
        # chat_id  uuid
        # request_id  uuid
        # text text(4096)
        # Мы используем chat_id как идентификатор пользователя

        chat_id = jdata['chat_id']
        input_text = jdata['text']

        logging.info('Input request: chat_id=%s request_id=%s text=%s', chat_id, jdata['request_id'], input_text)

        # обрезать слишком длинный текст?
        #if len(input_text) > 200:
        #    # TODO: сделать обрезание на пробельном символе
        #    input_text = input_text[:200]

        # Исходящий ответ должен содержать json-структуру с единственным полем text.
        # В это поле мы упакуем все накопившиеся реплики бота.
        output_text = '---тут ответ---'

        ERROR_REPLY = 'что-то тут у меня все сломалось :('

        try:
            response = requests.get(chatbot_url + '/' + 'push_phrase?user={}&phrase={}'.format(chat_id, input_text))
            if not response.ok:
                logging.error(response.error)
                output_text = ERROR_REPLY
            else:
                replies = []
                while True:
                    response = requests.get(chatbot_url + '/' + 'pop_phrase?user={}'.format(chat_id))
                    if response.ok:
                        reply = response.json()['reply']
                        if reply:
                            replies.append(reply)
                        else:
                            break
                    else:
                        logging.error(response.error)
                        replies.append(ERROR_REPLY)
                        break

            output_text = '\n'.join(replies)

        except Exception as ex:
            logging.error(ex)
            output_text = ERROR_REPLY

        logging.info('Output: chat_id=%s request_id=%s output_text=%s', chat_id, jdata['request_id'], output_text)

        return jsonify({'text': output_text})
    else:
        return Response(status=404)


@flask_app.route('/start_conversation', methods=["GET"])
def start_conversation():
    user_id = request.args.get('user')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность

    logging.debug('start_conversation user=%s', user_id)

    response = requests.get(chatbot_url + '/' + 'start_conversation?user={}'.format(user_id))
    if not response.ok:
        logging.error(response.error)
        return Response(status=404)
    else:
        return jsonify(response.json())


@flask_app.route('/push_phrase', methods=["GET"])
def push_phrase():
    phrase = request.args['phrase']
    user_id = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность

    logging.debug('push_phrase user=%s phrase=%s', user_id, phrase)

    response = requests.get(chatbot_url + '/' + 'push_phrase?user={}&phrase={}'.format(user_id, phrase))
    if not response.ok:
        logging.error(response.error)
        return Response(status=404)
    else:
        return jsonify(response.json())


@flask_app.route('/pop_phrase', methods=["GET"])
def pop_phrase():
    user_id = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность

    logging.debug('pop_phrase user=%s', user_id)

    response = requests.get(chatbot_url + '/' + 'pop_phrase?user={}'.format(user_id))
    if not response.ok:
        logging.error(response.error)
        return Response(status=404)
    else:
        return jsonify(response.json())


@flask_app.route('/cancel_all_running_items', methods=["GET"])
def cancel_all_running_items():
    user_id = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность

    logging.debug('cancel_all_running_items user=%s', user_id)

    response = requests.get(chatbot_url + '/' + 'cancel_all_running_items?user={}'.format(user_id))
    if not response.ok:
        logging.error(response.error)
        return Response(status=404)
    else:
        return jsonify(response.json())


if __name__ == '__main__':
    flask_app.run(debug=False, host='0.0.0.0', port=9000)
