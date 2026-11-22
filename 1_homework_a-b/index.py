import pandas as pd

# Напишите Python-скрипт, который:

# 1. Загружает experiment_data.csv и выводит базовую информацию (размеры групп, среднюю конверсию, частоту ошибок).

df = pd.read_csv('1_homework_a-b/experiment_data.csv', parse_dates=True)

print('Общий размер выборки', len(df))

print('Количество групп', len(df['variant'].value_counts(normalize=True)))

test_group = df[df['variant'] == 'test']
control_group = df[df['variant'] == 'control']

print('Размер тестовой группы', len(test_group))
print('Размер контрольной группы', len(control_group))

print(f'Средняя конверсия: {df['converted'].mean():.3f}')
print(f'Частота ошибок: {((df['error_flag'] == 1.0).sum() / len(df) * 100):.2f}%')


# 2. Проверяет SRM (Sample Ratio Mismatch), например, с помощью хи-квадрат теста.
#     Ожидаемое соотношение 50/50.
#     Если p-value < 0.05, вывести предупреждение «SRM обнаружен».

print(
    'Проверка соотношения 50/50, если False присутствует большое расхождение -',
    0.95 < (len(test_group) / len(control_group)) < 1.05
)

observed = [len(control_group), len(test_group)]
total_users = sum(observed)
expected = [total_users / 2, total_users / 2]

chi2_stat, p_val = scipy.stats.chisquare(f_obs=observed, f_exp=expected)

print(f"p-value: {p_val:.5f}")
if p_val < 0.05:
    print("SRM обнаружен")
else:
    print("SRM не обнаружен")

# 3. Проверяет баланс ковариаты pre_activity между группами.
#     Используйте t-test (или U-тест, если распределение не нормальное).
#     Если различия статистически значимы (p-value < 0.05), вывести предупреждение «Дисбаланс по pre_activity — возможен confounder».

_, p_norm_control = scipy.stats.shapiro(control_group['pre_activity'])
_, p_norm_test = scipy.stats.shapiro(test_group['pre_activity'])

print(f"p-value теста Шапиро-Уилка (Control): {p_norm_control:.5f}")
print(f"p-value теста Шапиро-Уилка (Test): {p_norm_test:.5f}")

is_normal = (p_norm_control >= 0.05) and (p_norm_test >= 0.05)

if is_normal:
    print("Распределения нормальные. Используем t-test...")
    stat, p_val = scipy.stats.ttest_ind(control_group['pre_activity'], test_group['pre_activity'], equal_var=False)
    test_name = "t-test"
else:
    print("Распределения ненормальные. Используем U-test Манна-Уитни...")
    stat, p_val = scipy.stats.mannwhitneyu(control_group['pre_activity'], test_group['pre_activity'], alternative='two-sided')
    test_name = "U-test"

print(f"p-value ({test_name}): {p_val:.5f}")
if p_val < 0.05:
    print("Дисбаланс по pre_activity — возможен confounder")
else:
    print("Дисбаланс по pre_activity не обнаружен")

# 4. Анализирует guardrail-метрику error_flag.
#     Посчитать частоту ошибок в контрольной и тестовой групп

from statsmodels.stats.proportion import proportions_ztest

# Считаем количество ошибок и общий размер групп
errors_control = control_group['error_flag'].sum()
errors_test = test_group['error_flag'].sum()

n_control = len(control_group)
n_test = len(test_group)

rate_control = errors_control / n_control
rate_test = errors_test / n_test

print(f"Частота ошибок в Control: {rate_control * 100:.2f}%")
print(f"Частота ошибок в Test: {rate_test * 100:.2f}%")

# Z-тест для двух долей
stat, p_val = proportions_ztest([errors_control, errors_test], [n_control, n_test])

print(f"p-value (Z-test): {p_val:.5f}")
if p_val < 0.05:
    print("Обнаружено значимое изменение частоты ошибок (guardrail-метрика нарушена)!")
else:
    print("Значимых изменений в частоте ошибок не обнаружено")

# 5. Принимает решение о валидности эксперимента на основе пунктов 2–4:
#     Если есть хотя бы одно предупреждение → эксперимент не валиден, вывод: «Остановить тест, исправить ошибки, перезапустить».
#     Если всё в порядке → провести стандартный анализ первичной метрики (конверсии) и сделать вывод о значимости.

# Определяем флаги проблем на основе рассчитанных p-value
srm_warning = (scipy.stats.chisquare([len(control_group), len(test_group)], [total_users/2, total_users/2]).pvalue < 0.05)

_, p_val_mw = scipy.stats.mannwhitneyu(control_group['pre_activity'], test_group['pre_activity'], alternative='two-sided')
confounder_warning = (p_val_mw < 0.05)

_, p_val_z = proportions_ztest([errors_control, errors_test], [n_control, n_test])
guardrail_warning = (p_val_z < 0.05)

# Принимаем решение о валидности
if srm_warning or confounder_warning or guardrail_warning:
    print("Решение: Остановить тест, исправить ошибки, перезапустить")
else:
    print("Эксперимент валиден. Проводим стандартный анализ конверсии...")
    # Тест на равенство конверсий
    conv_control = control_group['converted'].sum()
    conv_test = test_group['converted'].sum()
    stat_conv, p_val_conv = proportions_ztest([conv_control, conv_test], [n_control, n_test])
    print(f"p-value (конверсия): {p_val_conv:.5f}")
    if p_val_conv < 0.05:
        print("Результаты стат. значимы! Есть эффект от изменений.")
    else:
        print("Статистически значимых различий в конверсиях не обнаружено.")

# 6. Бонус (необязательно, но желательно):
#     Предложите метод исправления одной из проблем (например, взвешивание наблюдений для коррекции дисбаланса по pre_activity или удаление несостоятельных данных).
#     Продемонстрируйте, как изменится вывод после исправления (например, отфильтровав данные до сбалансированных по ковариате).
