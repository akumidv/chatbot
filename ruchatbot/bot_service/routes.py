# -*- coding: utf-8 -*-

"""
Привязка урлов к функциям сервиса чатбота https://github.com/Koziev/chatbot.
"""

from __future__ import print_function

from flask import request
from flask import render_template
from flask import redirect
from flask import jsonify

#from bot_service import flask_app
from .dialog_form import DialogForm
from .dialog_phrase import DialogPhrase
from .rest_service_core import flask_app


user_id = 'anonymous'


@flask_app.route('/start', methods=["GET"])
def start():
    # Чтобы заранее заставить бота загрузить все модели с диска.
    return redirect('/index')
    pass


@flask_app.route('/', methods=["GET", "POST"])
@flask_app.route('/index', methods=["GET", "POST"])
def index():
    # покажем веб-форму, в которой пользователь сможет ввести
    # свою реплику и увидеть ответ бота
    form = DialogForm()
    bot = flask_app.config['bot']

    # todo - сохранять и восстанавливать историю диалога через БД...
    DIALOG_HISTORY = 'dialog_history'
    if DIALOG_HISTORY not in flask_app.config:
        flask_app.config[DIALOG_HISTORY] = dict()

    if user_id not in flask_app.config[DIALOG_HISTORY]:
        flask_app.config[DIALOG_HISTORY][user_id] = []

    phrases = flask_app.config[DIALOG_HISTORY][user_id][:]

    if form.validate_on_submit():
        # Пользователь ввел свою реплику, обрабатываем ее.
        # flash('full_name={}'.format(form.full_name.data))
        utterance = form.utterance.data
        if len(utterance) > 0:
            phrases.append(DialogPhrase(utterance, user_id, False))
            bot.push_phrase(user_id, utterance)

    if request.method == 'GET':
        bot.start_conversation(user_id)

    while True:
        answer = bot.pop_phrase(user_id)
        if len(answer) == 0:
            break
        phrases.append(DialogPhrase(answer, 'chatbot', True))

    flask_app.config[DIALOG_HISTORY][user_id] = phrases

    return render_template('dialog_form.html', hphrases=phrases, form=form)


@flask_app.route('/start_conversation', methods=["GET"])
def start_conversation():
    user = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность
    bot = flask_app.config['bot']
    bot.start_conversation(user)
    return jsonify({'processed': True})


@flask_app.route('/push_phrase', methods=["GET"])
def push_phrase():
    phrase = request.args['phrase']
    user = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность
    bot = flask_app.config['bot']
    bot.push_phrase(user, phrase)
    return jsonify({'processed': True})


@flask_app.route('/pop_phrase', methods=["GET"])
def pop_phrase():
    user = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность
    bot = flask_app.config['bot']
    reply = bot.pop_phrase(user)
    return jsonify({'reply': reply})


@flask_app.route('/cancel_all_running_items', methods=["GET"])
def cancel_all_running_items():
    user = request.args.get('user', 'anonymous')
    bot_id = request.args.get('bot', None)  # TODO - использовать, когда появится мультипрофильность
    bot = flask_app.config['bot']

    bot.cancel_all_running_items(user)
    bot.reset_session(user)
    bot.cancel_all_running_items(user)
    bot.reset_usage_stat()
    bot.reset_added_facts()

    return jsonify({'processed': True})
