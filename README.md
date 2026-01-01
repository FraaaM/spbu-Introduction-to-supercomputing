# spbu-Introduction-to-supercomputing
## Задачи по OpenMP и MPI находятся в HPC-tasks-2025.pdf

## bash команды для работы на кластере при решении задач MPI:
```bash
scp -P 12345 your_src.cpp your_sh.sh your_username@cmmshq.ru:/home/your_username/your_folder/ # из локальной директории установка на кластер
scp -P 12345 your_username@cmmshq.ru:~/your_folder/your_results.csv . # C кластера загрузка в локальную директорию
ssh -p 12345 your_username@cmmshq.ru # Запуск кластера
chmod +x your_sh.sh # Права 
sed -i 's/\r$//' your_sh.sh #  форматирование (необязательно)
sbatch your_sh.sh # Запуск скрипта
squeue -u your_username # Проверка запущенных процессов
scancel process_name # Отмена процесса process_name
```

