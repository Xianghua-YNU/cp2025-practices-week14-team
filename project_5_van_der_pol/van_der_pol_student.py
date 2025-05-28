import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from typing import Tuple, Callable, List, Optional

def van_der_pol_ode(t: float, state: np.ndarray, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """Van der Pol振子的一阶微分方程组。
    
    Args:
        t: 当前时间
        state: 状态向量 [x, v]
        mu: 非线性阻尼系数
        omega: 固有频率
        
    Returns:
        导数向量 [dx/dt, dv/dt]
    """
    x, v = state
    return np.array([v, mu*(1 - x**2)*v - omega**2*x])

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
             dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """使用solve_ivp求解常微分方程组。
    
    Args:
        ode_func: 微分方程函数，形式为 func(t, state, *args)
        initial_state: 初始状态数组
        t_span: 时间范围 (t0, t1)
        dt: 时间步长（输出解的时间间隔）
        **kwargs: 传递给ode_func的额外参数
        
    Returns:
        t: 时间数组
        states: 状态数组，每行对应一个时间点的状态
    """
    n_points = int((t_span[1] - t_span[0]) / dt) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """绘制状态变量的时间演化图。
    
    Args:
        t: 时间数组
        states: 状态数组
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """绘制相空间轨迹图。
    
    Args:
        states: 状态数组
        title: 图表标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_limit_cycle(t: np.ndarray, states: np.ndarray, 
                       skip_ratio: float = 0.5) -> Tuple[float, float]:
    """分析极限环的特征（振幅和周期）。
    
    Args:
        t: 时间数组
        states: 状态数组
        skip_ratio: 跳过初始数据比例
        
    Returns:
        振幅和周期的元组
    """
    skip = int(len(states) * skip_ratio)
    x = states[skip:, 0]
    t_trimmed = t[skip:]
    
    # 使用scipy的峰值检测
    peaks, _ = find_peaks(x)
    
    if len(peaks) < 2:
        return np.nan, np.nan
    
    peak_values = x[peaks]
    peak_times = t_trimmed[peaks]
    
    amplitude = np.mean(peak_values)
    periods = np.diff(peak_times)
    period = np.mean(periods)
    
    return amplitude, period

def main():
    # 基础参数设置
    mu = 1.0
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基础实现
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'Van der Pol Oscillator Time Evolution (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [1.0, 2.0, 4.0]
    for mu_val in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu_val, omega=omega)
        plot_time_evolution(t, states, f'Van der Pol Oscillator Time Evolution (μ={mu_val})')
        amplitude, period = analyze_limit_cycle(t, states)
        print(f'μ = {mu_val}: Amplitude ≈ {amplitude:.3f}, Period ≈ {period:.3f}')
    
    # 任务3 - 相空间分析
    for mu_val in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu_val, omega=omega)
        plot_phase_space(states, f'Van der Pol Phase Space (μ={mu_val})')

if __name__ == "__main__":
    main()
