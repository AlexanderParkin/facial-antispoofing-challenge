Команды создания нужного окружения:

conda create -n meta python=3.6.6

source activate meta

conda install pytorch torchvision -c pytorch # cuda90 - important

conda install jupyter

conda install -c anaconda pandas 




Список TODO:
1. <b>DONE</b> Файл генерации opts. 
2. Вынести transforms из opts
3. <b>DONE</b> Добавить missing data trainer) 
4. Визуализация в Visdom
5. Логгирование с названиями метрик в консоль
6. Нормальные комменты у кода
7. Профайлинг узких мест по времени 
8. <b>DONE</b> Отдельный скрипт инференса