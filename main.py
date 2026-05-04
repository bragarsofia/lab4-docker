# Модель: Оптимальне керування процесом очищення водойми
# Автор: Брагар Софія в групі з Маляренко Анастасією, група АІ-233
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class WaterOptimizer:
    def __init__(self, V, Q, k, C_initial, C_target, Cin_max):
        """Ініціалізація параметрів системи"""
        self.V = V; self.Q = Q; self.k = k
        self.C_initial = C_initial
        self.C_target = C_target
        self.Cin_max = Cin_max

    def cstr_ode(self, t, C, Cin_profile, t_points):
        """Права частина ЗДР (динаміка концентрації)"""
        u_t = np.interp(t, t_points, Cin_profile)
        dCdt = (self.Q / self.V) * u_t - (self.Q / self.V + self.k) * C
        return dCdt

    def objective_minimum_effort(self, Cin_profile, T_fix, N_steps):
        """Функція цілі: Мінімізація витрат зусиль"""
        t_points = np.linspace(0, T_fix, N_steps)
        sol = solve_ivp(
            self.cstr_ode, (0, T_fix), [self.C_initial],
            args=(Cin_profile, t_points), method='RK45', t_eval=t_points
        )
        C_final = sol.y[0, -1]
        penalty = 0 if C_final <= self.C_target else (C_final - self.C_target)**2 * 1e5
        dt = T_fix / (N_steps - 1)
        effort_cost = np.sum(Cin_profile**2) * dt
        return effort_cost + penalty

    def find_optimal_control(self, T_fix, N_steps=50):
        """Запуск оптимізації SLSQP"""
        initial_guess = np.ones(N_steps) * self.Cin_max
        bounds = [(0, self.Cin_max)] * N_steps
        result = minimize(
            self.objective_minimum_effort, initial_guess,
            args=(T_fix, N_steps), method='SLSQP', bounds=bounds
        )
        return result.x, result.fun

if __name__ == "__main__":
    # Отримання даних із Docker ENV
    student_name = os.getenv("STUDENT_NAME", "Брагар Софія")
    group_name = os.getenv("GROUP", "АІ-233")
    mode = os.getenv("MODE", "не визначено")

    print("="*50)
    print(f"ЛАБОРАТОРНА РОБОТА №4: КОНТЕЙНЕРИЗАЦІЯ")
    print(f"Модель: Оптимальне керування процесом очищення водойми")
    print(f"Виконавець: {student_name}")
    print(f"Група: {group_name}")
    print(f"Варіант: №1 (Режим: {mode})")
    print("="*50)

    # Запуск розрахунку
    optimizer = WaterOptimizer(V=1000, Q=50, k=0.1, C_initial=10, C_target=2, Cin_max=5)
    print(f"Проведення розрахунків для режиму {mode}...")
    
    try:
        u_opt, cost = optimizer.find_optimal_control(T_fix=10, N_steps=20)
        print("\n[РЕЗУЛЬТАТИ]")
        print(f"Мінімальні витрати енергії: {cost:.4f}")
        print(f"Статус: Розрахунок завершено успішно!")
    except Exception as e:
        print(f"Помилка: {e}")
    print("="*50)
