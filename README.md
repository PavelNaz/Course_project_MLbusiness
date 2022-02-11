# Course_project_MLbusiness
 
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy API: flask Данные с kaggle - https://www.kaggle.com/shubh0799/fake-news

Задача: предсказать по опубликовано тексту является ли он фейком или нет (поле label). Бинарная классификация

Используемые признаки:

title
text
Преобразования признаков: tfidf

Модель: GradientBoostingClassifier

Описание шагов
- step_1 - загрузка данных, сбор Pipeline и обучение модели;
- step_2 - проверка работоспособности и нашего Pipeline;
- step_3 - Запрос к сервису;
- run_server.py - наш сервис, который запускаем
Запуск
- Создайте пустой каталог, в котором будут содержаться файлы проекта, а внутри него virtualenv окружение:
 $ mkdir my_project
 $ cd my_project
 $ virtualenv venv или если у вас анаконда (conda create -n venv python=3.7)
- Активируйте окружение и установите в него requirements.txt с пакетами:
 $ source venv/bin/activate или conda activate venv 
 $ pip install -r requirements.txt # (находясь в корне проекта конечно же при этом)
- запустить наш сервис:
 $ python run_server.py
- в ноутбуке step_3 можно приступать к работе с нашим сервисом
