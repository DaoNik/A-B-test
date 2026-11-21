import numpy as np
import pandas as pd

# СГЕНЕРИРУЙТЕ СВОЙ SEED 
MY_SEED = np.random.randint(5000000, size=1)[0]
np.random.seed(MY_SEED)

# СГЕНЕРИРУЙТЕ ЧИСЛО УЧАСТНИКОВ ЭКСПЕРИМЕНТА
n_users = np.random.randint(1000, 10000, size=1)[0]

# Шаг 1: создаём ковариату pre_activity (логинормальное распределение)
pre_activity = np.random.exponential(scale=10, size=n_users).astype(int)

# Шаг 2: НАРУШЕНИЕ 1 — SRM (Sample Ratio Mismatch)
# Вместо 50/50 делаем 40/60
variant = np.random.choice(['control', 'test'], size=n_users, p=[0.4, 0.6])

# Шаг 3: НАРУШЕНИЕ 2 — confounder (рандомизация зависит от pre_activity)
# Для тестовой группы увеличиваем вероятность назначения для активных пользователей
# (это нарушает независимость)
prob_test = 0.4 + 0.3 * (pre_activity / pre_activity.max())
variant2 = np.where(np.random.random(n_users) < prob_test, 'test', 'control')

# Комбинируем оба нарушения (можно взять одно из двух, но для задачи оставим оба)
# Для простоты выберем основное назначение через variant2 (confounder)
variant = variant2

# НАРУШЕНИЕ 3 — дисбаланс guardrail (ошибки чаще в тесте)
# В тесте ошибки встречаются в 2 раза чаще, хотя истинного эффекта нет
error_flag = np.zeros(n_users)
error_flag[variant == 'control'] = np.random.binomial(1, 0.01, sum(variant == 'control'))
error_flag[variant == 'test'] = np.random.binomial(1, 0.02, sum(variant == 'test'))

# Конверсия не зависит от варианта, но зависит от pre_activity
conversion_prob = 0.05 + 0.01 * (pre_activity / pre_activity.max())
converted = np.random.binomial(1, conversion_prob, n_users)

# Шаг 4: Генерация дат (например, за 14 дней, с возможным смещением)
start_date = pd.Timestamp('2026-05-01')
dates = [start_date + pd.Timedelta(days=np.random.randint(0, 14)) for _ in range(n_users)]

df = pd.DataFrame({
    'user_id': range(n_users),
    'variant': variant,
    'converted': converted,
    'error_flag': error_flag,
    'pre_activity': pre_activity,
    'date': dates})

df.to_csv('experiment_data.csv', index=False)
print("Данные сохранены.Идентифицируйте проблемы")
