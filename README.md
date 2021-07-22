# Вопросно-ответная диалоговая система (чатбот)

## Краткое описание

Чатбот - это набор инструментов, позволяющих организовывать диалоговые сессии примерно такого вида:

```
B:> Здравствуй
H:> привет
B:> Привет, давай познакомимся?
H:> давай
B:> Хорошо
B:> Как тебя зовут?
H:> илья
B:> меня зовут Вика
B:> Где ты живешь?
H:> в москве
B:> Тебе нравится в Москве?
H:> да
B:> Чем ты занимаешься?
H:> работаю на стройке
B:> Это опасная работа.
B:> Чем ты увлекаешься?
H:> компьютерными играми
B:> Теперь мы знакомы
B:> Я люблю компьютерные игры.
```

В этом примере реплики чатбота отмечены символами B:>, а реплики человека - символами H:>.

Данный чатбот архитектурно сочетает два подхода.

Во-первых, retrieval-based архитектура: ответ на заданный вопрос ищется в базе знаний с помощью набора NLP моделей.

Во-вторых, rule-based подход: чатбот может генерировать реплики и управлять диалогом помощью описанный вручную
**правил, сценариев и веб-форм** - см. далее описание файла [rules.yaml](./data/rules.yaml).

База знаний чатбота состоит из двух больших частей - базы фактов и FAQ. В [базе фактов](https://github.com/Koziev/chatbot/blob/master/data/profile_facts_1.dat)
ищется факт, на основе которого можно сформулировать ответ, даже если текст ответа в явном виде не содержится
в факте. В [FAQ](https://github.com/Koziev/chatbot/blob/master/data/faq2.txt) ищутся готовые ответы на типовые
вопросы, при этом текст ответа выдается собеседнику без изменений. Поиск информации в базе фактов осуществляется
моделью релевантности предпосылки и вопроса. Подбор подходящей записи в FAQ выполняется помощью детектора
синонимичности. Обе эти модели обучаются на больших датасетах.

Правила и вербальные формы описываются вручную, но также опираются на несколько NLP моделей. Модель синонимичности
и классификатор интента позволяют выбирать подходящее правило. Named Entity Recognition модуль извлекает из реплики
человека необходимую ключеую информацию.  

По умолчанию бот отвечает пассивно, не пытаясь задавать уточняющие вопросы и т.д. Процедурные средства
(правила, сценарии, веб-формы) могут сделать бота более проактивным, он будет задавать свои вопросы собеседнику,
активно пополняя свою базу знаний. Например, без сценариев реакция бота на вопрос "Как тебя зовут" выглядит так:

```
H:> Как тебя зовут?
B:> вика
```

То есть чатбот нашел в своей [базе фактов](https://github.com/Koziev/chatbot/blob/master/data/profile_facts_1.dat)
релевантную информацию и сгенерировал ответ. А с использованием сценария
диалог выглядит примерно так:

```
B:> Здравствуй
H:> как тебя зовут
B:> меня зовут Вика
B:> А тебя как зовут?
H:> Артур
B:> Редкое имя.
```

Уникальная особенность базы фактов в чатботе состоит в том, что бот пополняет ее в ходе диалога. Например,
продолжение вышеуказанного примера может выглядеть так:

```
H:> а скажи, как меня зовут?
B:> Артур 
```

То есть чатбот запомнил, что ему сказало собеседник - свое имя, и включил этот новый факт в стандартный
процесс генерации ответа.

## Запуск

Самый простой способ запуска чатбота - выкачать и запустить docker-образ [текущего релиза](https://github.com/Koziev/chatbot/releases).
Допустим, chatbot.tar.gz это скачанный файл, тогда для запуска нужно выполнить команды:

```
docker image load -i chatbot.tar.gz
docker run -ti -e PYTHONIOENCODING=utf-8 chatbot
```

Текущая версия чатбота не оптимизирована и достаточно долго загружает
файлы словарей и моделей. После появления приглашения можно ввести тестовые вопросы:

```
Привет, как тебя зовут?
Наверное, ты робот, да?
Что ты любишь делать?
Ты в шахматы умеешь играть?
А в шашки?
Ты знаешь, что такое белый карлик?
```

Можно запустить веб-сервис бота и общаться в браузере. Для этого нужно выполнить команду:

```
docker run -p 9001:9001 -it chatbot bash -c "/chatbot/scripts/flask_bot.sh"
```

После того, как сервис внутри докер-контейнера стартует (это займет минут 5) и выведет:

```
werkzeug  -  * Running on http://0.0.0.0:9001/ (Press CTRL+C to quit)
```

можно открыть в браузере адрес http://127.0.0.1:9001 и вводить вопросы к чатботу.
 
Запуск чатбота в Telegram:

```
docker run -it chatbot bash -c "/chatbot/scripts/tg_bot.sh"
``` 

После старта появится приглашение ввода токена, который следует получить у @botfather в Telegram.
 


## Кастомизация чатбота, константы профиля

Используемые базы знаний и FAQ, а также наборы правил ведения диалога указываются
в профиле, который загружается при старте экземпляра бота. В скрипте console_bot.sh
можно увидеть указание на тестовый профиль [profile_1.json](https://github.com/Koziev/chatbot/blob/master/data/profile_1.json),
позволяющий боту отвечать на несколько простых вопросов. В этом профиле в качестве базы знаний указан файл [profile_facts_1.dat](https://github.com/Koziev/chatbot/blob/master/data/profile_facts_1.dat).
Формат этого файла описан в шапке файла.


Среди разных фактов там можно увидеть запись:

```
меня зовут $name_nomn
```

Конструкция $name_nomn означает, что в строку при загрузке чатбота будет подставлена
*константа* с именем name_nomn, определенная в файле profile_1.json в разделе constants:

```
	"constants": {
		"gender": "ЖЕН",
		"name_nomn": "Вика"
	}
```

Так как бота может встречаться в нескольких местах (база фактов, FAQ, правила), то удобнее
задать имя в одном месте, в файле профиля. 

Когда чатбот обрабатывает вопрос "Как тебя зовут?", он определяет, что этот факт наиболее
релевантен для ответа на заданный вопрос, и далее запускает процедуру построения
ответа. Само имя "Вика" нигде не "зашито" в языковых моделях. Поэтому для его смены не нужно переобучать
нейросетки, а достаточно отредактировать данную запись.

Вторая константа с именем "gender" определяет грамматический род для бота, в данном случае женский. В том же файле
фактов можно найти такую запись:

```
Я $chooseAdjByGender(нужен, нужна), чтобы отвечать на вопросы посетителей чата
```

Конструкция $chooseAdjByGender(нужен, нужна) позволяет выбрать одно из перечисленных слов, фильтруя
их по константе грамматического рода. Таким образом, реплики бота становятся более релевантными
"биологической" природе бота.

Все вышесказанное справедливо и для записей в [FAQ](data/faq2.txt), и для [правил скриптования бота](data/rules.yaml).

При добавлении новых фраз в вышеуказанные файлы следует по возможности воздерживаться
от использования лексики, неизвестной языковым моделям чатбота.
С определенными оговорками, список слов в файле [tmp/dataset_words.txt](https://github.com/Koziev/chatbot/blob/master/tmp/dataset_words.txt)
известен чатботу и модет использоваться.

Остальные правила для движка чатбота собраны в файле [data/rules.yaml](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml).


## Правила для чатбота

Правила собраны в файле [data/rules.yaml](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml) с форматом YAML.
Смотрите комментарии в файле, поясняющие структуру правил, а также пояснения
в разделе "Порядок применения правил" далее. Привязка набора правил к экземпляру
бота выполняется в профиле - текстовом файле типа [profile_1.json](https://github.com/Koziev/chatbot/blob/master/data/profile_1.json).

**Вербальные формы** задают перечень **слотов**, которые пользователь должен заполнить своими ответами. Каждый
слот задается вопросом, который бот задаст, если соответствующая информация отсутствует в стартовой реплике
собеседника. Например, для формы заказа еды нужно, чтобы пользователь указал, **что** он хочет заказать, **сколько**
порций и **когда** нужно привезти заказ. Вербальная форма активируется правилом, которое реагирует на ***намерение** (aka intent)
или текст реплики. К примеру, правило-триггер может реагировать на наличие ключевого слова *заказать*. Если пользователь
скажет *"хочу заказать пиццу"*, то запустившаяся вербальная форма найдет в этой фразе заполнение для слота "какую еду",
а для слотов "сколько порций" и "когда доставить" информация будет запрашиваться заданными уточняющими вопросами. После
того, как все слоты вербальной формы будут заполнены, будет активирован обработчик, например заданный пользовательский
код. Соответствующая диалоговая сессия выглядит примерно так:
 
 ```
B:> Добрый день
H:> хотелось бы заказать что-нибудь покушать
B:> Что заказать?
H:> пиццу с ананасами
B:> Сколько порций?
H:> две
B:> Заказываю: что="пиццу с ананасами", сколько="две"
```

Технически извлечением информации для слотов заведует [Named Entity Recognition модуль](https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/entity_extractor.py) со своей нейросетевой
классификационной моделью, тренируемой на расширенной версии [датасета](https://github.com/Koziev/chatbot/blob/master/data/entity_extraction.txt).
 
Пример вербальной формы можно найти в файле [rules.yaml](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml)
поиском по подстроке "form".
 
**Сценарии** описывают примерный перечень вопросов, который бот будет задавать после какой-либо реплики
собеседника с заданным интентом или текстом. Пример работы простого сценария:

```
B:> Добрый день
H:> как тебя зовут
B:> меня зовут Вика
B:> А тебя как зовут?
H:> Артур
B:> Редкое имя.
```

Особенность сценариев в том, что перед тем, как задать 
прописанный в сценарии вопрос, чатбот сначала проверит: не знает ли он уже ответ на этот вопрос. Если у бота уже
есть нужная информация, вопрос он задавать не будет. Например, если собеседник ранее уже сообщил свое имя и оно
было запомнено в базе фактов, то прописанный в сценарии вопрос "как тебя зовут" задавать не надо. Такая "рефлексия"
делает диалог с чатботом менее механическим и более живым. Пример сценариев можно найти в файле файле [rules.yaml](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml)
поиском по подстроке "scenario".
 
## База знаний

Знания функционально разделены на 2 части.

FAQ-правила - состоят из пар "вопрос - ответ". Для удобства обработки
перефразировок вопросов для одного ответа может быть несколько. Когда
движок бота обрабатывает вопрос собеседника, он ищет среди FAQ-правил
наиболее близкий опорный вопрос. Если поиск удался, то в качестве ответной
реплики бота будет выдан текст из этого FAQ-правила. Сопоставление
опорных вопросов и запроса собеседника выполняется с помощью модели
синонимичности. В демо-версии чатбота FAQ-правила собраны в файле
[data/faq2.txt](https://github.com/Koziev/chatbot/blob/master/data/faq2.txt).
Для примера, введите вопрос "Что такое белые карлики" и бот выдаст
соответствующую инфомацию:

```
H:> что такое белые карлики
B:> Белые карлики — проэволюционировавшие звёзды с массой, не превышающей
предел Чандрасекара
```

F-правила, или просто факты, представляют из себя одиночные предложения,
описывающие элементарные факты о самом чатботе, собеседнике или окружении.
Получив вопрос собеседника, чатбот ищет в этой базе факт, максимально релевантный
заданному вопросу (сравни с FAQ-правилами). Если такой факт найден, то
он далее поступает в [движок генерации ответа](https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/answer_builder.py). Сопоставление
вопроса собеседника и предпосылок производится с помощью модели
релевантности. В демо-версии чатбота факты собраны в файле 
[profile_facts_1.dat](https://github.com/Koziev/chatbot/blob/master/data/profile_facts_1.dat).
К примеру, ответ на вопрос "Как тебя зовут?" подразумевает поиск 
соответствующего факта - см. абзац про кастомизацию бота.


## Порядок применения правил и моделей

1) Если есть история диалога (>1 реплики), то реплика собеседника
прогоняется через [модель интерпретации](https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/nn_interpreter_new2.py) для восстановления полной фразы,
раскрытия анафоры, гэппинга, эллипсиса и т.д.

2) Среди comprehension правил ищется достаточно близкий вариант фразы
в if блоке. Если нашлось, то вместо исходной фразы дальше будет
обрабатываться then-фраза из найденного правила. Таким образом
выполняется некоторая нормализация фраз собеседника. (deprecated: этот шаг, возможно, будет удален в будущем).

3) Определяется intent с помощью обученного на датасете [data/intents.txt](https://github.com/Koziev/chatbot/blob/master/data/intents.txt)
классификатора (см. далее). Также определяется сентимент, оскорбительность, направленность реплики. [Правила](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml) могут срабатывать
на определенную категорию. Например, правило для интента 'кто_я' запускает целый сценарий для "знакомства".

4) Определяется грамматическая модальность - является ли реплика вопросом,
утверждением или приказом. В текущей версии считается, что все вопросы к боту, выраженные сказуемым во 2 лице, являются вопросами,
даже если символ "?" не использован: "кто ты".

5) Для приказов: пытаемся найти правило для обработки
(секция rules в [rules.yaml](https://github.com/Koziev/chatbot/blob/master/data/rules.yaml)) и выполняем его. При поиске используется либо
определенный intent (if-часть содержит ключевое слово intent), либо
проверяется синонимичность с помощью модели синонимичности.
Если правило не найдено, то вызывается дефолтный обработчик -
пользовательская функция, зарегистрированная в on_process_order.
Если и он не обработал приказ, то будет сказана фраза "unknown_order"
в rules.yaml

6) Для утверждений: пытаемся найти правило обработки (секция rules
в rules.yaml) и выполнить его. Далее, факт сохраняется в базе знаний.
Наконец, пытаемся найти smalltalk-правило: это правило в группе rules (rules.yaml),
в котором опорная часть (if) и результативная часть (then) заданы с ключевым
словом text. Ищется правило, в котором опорная часть максимально синонимична
входной фразе, если найдено - чатбот скажет фразу, которая указана в then-ветке.

7) Для вопросов: сначала проверяется, нет ли похожего (модель синонимичности)
вопроса среди FAQ-правил (файл faq2.txt). Если есть - выдается содержимое
найденного FAQ-правила. Иначе начинается процедура генерации ответа.
С помощью модели релевантности (см. отдельный раздел про ее дообучение
и валидацию) ищутся максимальной релевантные предпосылки в файлах premises*.txt.
Если не найдена достаточно релевантная предпосылка, то выдается фраза
"no_relevant_information" из rules.yaml.

## Переобучение модели релевантности

При добавлении новых фактов в базу знаний может возникнуть ситуация,
что модель релевантности не знакома с новой лексикой и сильно ошибается
при поиске подходящего факта. В этом случае модель релевантности можно
переобучить, добавив новые сэмплы в обучающий датасет. Обучающий датасет - это текстовый
tab-separated файл [premise_question_relevancy.csv](https://github.com/Koziev/chatbot/blob/master/data/premise_question_relevancy.csv).
В колонке premise находятся предпосылки (факты), question - вопросы. Колонка
relevance содержит 1 для нелевантных пар, 0 для нерелевантных. Таким
образом, чтобы модель считала предпосылку и вопрос релевантными,
надо добавить к этому датасету запись с relevance=1. Следует избегать
добавления повторов, так как это будет приводить к искажению оценок
точности при обучении.

После изменения файла premise_question_relevancy.csv нужно запустить
обучение скриптом [train_lgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/train_lgb_relevancy.sh). Обучение идет
примерно полчаса. В результате в каталоге .../tmp будут созданы новые
файлы lgb_relevancy.*, содержащие правила модели релевантности.

## Контроль качества модели релевантности

Любые ошибки при работе модели релевантности негативно сказываются
на общем качестве диалогов, поскольку многие другие части чатбота
используют результаты выбора предпосылок из базы знаний в качестве
входной информации. Чтобы контролировать качество этой модели,
желательно верифицировать ее работу на тестовых вопросах и наборе
тестовых предпосылок. Для выполнения этой верификации мы используем
простой консольный скрипт [query2_lgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/query2_lgb_relevancy.sh).
Он загружает текущую обученную модель релевантности и список предпосылок из базы знаний
(файлы ../data/premises*.txt) и тренировочного датасета
[premise_question_relevancy.csv](https://github.com/Koziev/chatbot/blob/master/data/premise_question_relevancy.csv).
Затем с консоли вводится проверочный вопрос, модель вычисляет его релевантность по всем
предпосылкам и выводит список из нескольких самых релевантных. Если
в этом списке есть явно нерелевантные предпосылки с высокой оценкой
(допустим, выше 0.5), то есть смысл добавить такие предпосылки с
вопросом в качестве негативных примеров в датасет
premise_question_relevancy.csv и переобучить модель релевантности,
запустив скрипт [train_lgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/train_lgb_relevancy.sh).


## Модель определения intent'а

С помощью модели intent'а можно присвоить фразе собеседника
одну метку из набора возможных и далее обрабатывать фразу
с учетом этой метки правилами (раздел rules в rules.yaml).

Данные для обучения находятся в текстовом файле data/intents.txt
Его формат прост и описан в комментарии в начале файла.

После модификации этого файла необходимо собрать датасет с помощью
PyModels/preparation/prepare_intents_dataset.py, затем
запустить обучение классификатора с помощью scripts/train_intent_classifier.sh


## Отличительные особенности чатбота

1) Новая информация (факты), сообщаемые пользователем, могут немедленно добавляться
в базу знаний и использоваться в последующих ответах наряду с остальными фактами. Именно так
в базу знаний бота попадает имя собеседника, которое он сообщает. Переобучение моделей
при этом не требуется.

2) Сессионность: сообщаемые пользователем факты сохраняются в отдельном разделе
базы знаний, привязанном к пользователю, так что сессии для разных пользователей
могут оперировать противоречащей или конфиденциальной информацией.

3) Ориентированность на клиент-серверный механизм; способ доставки сообщений отвязан от
ядра системы и может быть реализован в виде текстовой консоли, бота для мессенджера,
браузерного клиента для веб-сервиса и т.д.

4) Ответы чатбота по умолчанию не программируются скриптами или аналогичными средствами, а генерируются
обучаемыми моделями. В необходимых случаях используются нейросетевые deep learning модели,
а также классические NLP алгоритмы.

Специальная модель [интерпретации ответов](https://github.com/Koziev/chatbot/blob/master/ruchatbot/trainers/nn_interpreter_new2.py) собеседника позволяет чатботу понимать неполные реплики,
опираясь на контекст беседы, и далее учитывать их как обычные факты:

```
B:> Здравствуй
H:> как тебя зовут?
B:> кеша
B:> А как тебя зовут?
H:> илья
B:> Приятно познакомиться.
H:> как меня зовут?
B:> илья
```

В данном примере чатбот задает вопрос "А как тебя зовут", рассчитывая заполнить пробел в своей базе
знаний. Собеседник сообщает свое имя, и его ответ интерпретируется как "Меня зовут Илья".
Эта реплика рассматривается как полноценный факт и запоминается в базе знаний, что видно
на последних двух репликах диалога, где собеседник спрашивает у бота свое имя и получает его. Раскрытие
реплик собеседника облегчает работу пайплайна, так как не нужно "вручную" учитывать контекст для задания
реакции на фразу "нет", к примеру. Хотя при необходимости можно опираться именно на исходный ответ
собеседника, до его раскрытия.


## Консольный фронтенд для бота

Реализован в файле [console_chatbot.py](https://github.com/Koziev/chatbot/blob/master/PyModels/console_chatbot.py).
Запуск под Linux выполняется скриптом scripts/console_bot.sh

![Console frontend for chatbot](chatbot-console.PNG)

Это отладочная консоль, в которую помимо реплик чатбота выводятся также различные диагностические сообщения.


## Технические подробности реализации

Список моделей:

Определение синонимии фраз [nn_synonymy_detector.py](https://github.com/Koziev/chatbot/tree/master/PyModels/bot/nn_synonymy_detector.py)  
Интерпретация реплики собеседника (раскрытие анафоры, элипсиса, гэппинга, дополнение ответа etc) [nn_interpreter_new2.py](https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/nn_interpreter_new2.py)  
Определение релевантности предпосылки и вопроса [lgb_relevancy_detector.py](https://github.com/Koziev/chatbot/tree/master/PyModels/bot/lgb_relevancy_detector.py)  
Генерация текста ответа с помощью seq2seq нейросетки [train_nn_seq2seq_pqa_generator.py](https://github.com/Koziev/chatbot/blob/master/ruchatbot/trainers/train_nn_seq2seq_pqa_generator.py)
Посимвольное встраивание слово в вектор фиксированной длины [wordchar2vector_model.py](https://github.com/Koziev/chatbot/tree/master/PyModels/bot/wordchar2vector_model.py)  
Определение достаточности набора предпосылок для генерации ответа [nn_enough_premises_model.py](https://github.com/Koziev/chatbot/tree/master/PyModels/bot/nn_enough_premises_model.py)
NER для некоторых типов сущностей [entity_extractor.py](https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/entity_extractor.py)

Набор моделей и конкретная реализация могут сильно меняться по мере развития проекта,
поэтому список является не окончательным.

Описание тренировки и использования модели посимвольного встраивания слов
смотрите на [отдельной странице](./PyModels/trainers/README.wordchar2vector.md).

Также доступно [описание модели для определения релевантности факта и вопроса](README.relevance.md).
