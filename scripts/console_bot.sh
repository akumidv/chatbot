# profile_1.json это файл с настройками бота; там задаются пути к файлам с фактами и т.д.
# флаг --debugging включит детальное логирование
PYTHONPATH=.. python3 ../ruchatbot/frontend/console_chatbot.py --profile ../data/profile_1.json --data_folder ../data --models_folder ../tmp --w2v_folder ../tmp --tmp_folder ../tmp
