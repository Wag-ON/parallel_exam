## Программа для экзамена "Многопроцессорное программирование и оптимизация программ"

### Описание программы.
Цель программы состоит в нахождении оптимального многоугольного контура, максимально точно соответствующего области на бинарной маске.

На вход программе подается набор бинарных масок с очертаниями крыш домов, найденных нейронной сетью.

Программа преобразует каждую область произвольной формы в многоугольник с прямыми углами, который будет наиболее точно соответствовать форме здания. 

### Описание метода
Каждая область обрабатывается собственноручно разработанным алгоритмом:
- На область накладываются варианты прямоугольной сетки с различными параметрами
- Для каждого варианта считается метрика IOU (Intersection over Union)
- Выбирается вариант с наибольшей метрикой
- Новая область отрисовывается на месте старой

Варианты прямоугольной сетки применяются поочередно на каждую область,
результат обработки одной области совершенно не зависит от результата обработки другой, что позволяет применить распараллеливание задач.
Каждая область отдается на обработку в один поток.

### Установка и настройка
1. Скачать репозиторий
2. В корне проекта запустить команду `pip install -e .`

    Установятся все пакеты, перечисленные в `requirements.txt`

### config.py
- `DIR_WITH_DATA` - путь до папки с бинарными масками для обработки
- `OUTPUT_DIR` - путь до папки, в которой будут сохранены новые маски
- `OUTPUT_EXTENTION` - расширение выходных файлов
- `THREADS` - количество потоков, используемых при обработке

 
