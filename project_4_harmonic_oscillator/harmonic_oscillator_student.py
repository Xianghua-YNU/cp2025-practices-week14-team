import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    # TODO: 实现简谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x
    x, v = state
    return np.array([v,-omega**2*x])
def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现非谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x^3
    return np.array([v,-omage**2*x**3])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # TODO: 实现RK4方法
    k1=ode_func(state,t,**kwargs)
    k2=ode_func(state+0.5*dt*k1,t+0.5*dt,**kwargs)
    k3=ode_func(state+0.5*dt*k2,t+0.5*dt,**kwargs)
    k4=ode_func(state+dt*k3,t+dt,**kwargs)
    return state+(dt/6.0)*(k1+2*k2+2*k3+k4)
    

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
    t_start,t_end=t_span
    t=np.arange(t_state,t_end+dt,dt)
    states=np.zeros((len(t),len(initial_state)))
    states[0]=initial_state
    for i in range(1,len(t)):
        states[i]=rk4_step(ode_func,states[i-1],t[i-1],dt,**kwargs)
    return t,states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure(figsize=(10,6))
    plt.plot(t,states[:,0],label='Position x(t)')
    plt.plot(t,states[:,1],label='Velocity v(t)')
    plt.xlable('Time t')
    plt.ylable('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure(figsize=(8,8))
    plt.plot(states[:,0],states[:,1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # TODO: 实现周期分析
    x=states[:,0]
    peaks=[]
    for i in range(1,len(x)-1):
        if x[i]>x[i-1] and x[i]>x[i+1]:
            peaks.append(t[i])
    if len(peaks)<2:
        return np.nan

    #计算相邻峰值之间的时间差的平均值
    periods=np.diff(peaks)
    return np.mean(periods)


def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    omega=1.0
    t_span=(0,50)
    dt =0.01
    
    # TODO: 任务1 - 简谐振子的数值求解
    # 1. 设置初始条件 x(0)=1, v(0)=0
    # 2. 求解方程
    # 3. 绘制时间演化图
    initial_state=np.array([1.0,0.0])
    t,states = solve_ode(harmonic_oscillator_ode,initial_state,t_span,dt,omega=omega)
    plot_time_evolution(t,states,'Time Evolution of Harmonic Oscillator')
    period = analyze_period(t,states)
    print(f'Harmonic Oscillator Period:{period:.4f}(Theoretical:{2*np.pi/omega:.4f})')
    
    
    # TODO: 任务2 - 振幅对周期的影响分析
    # 1. 使用不同的初始振幅
    # 2. 分析周期变化
    amplitudes = [0.5,1.0,2.0]
    periods = []
    for A in amplitudes:
        initial_state = np.array([A,0.0])
        t,states = solve_ode(harmonic_oscillator_ode,initial_states,t_span,dt,omeaga=omega)
        period = analyze_period(t,states)
        periods.append(period)
        print(f'Amplitude{A}:Period = {period:.4f}')
    # TODO: 任务3 - 非谐振子的数值分析
    # 1. 求解非谐振子方程
    # 2. 分析不同振幅的影响
    for A in amplitudes:
        initial_state = np.array([A,0.0])
        t,state = solve_ode(hatmonic_oscillator_ode,initial_state,t_span,dt,omega=omega)
        period = analyze_period(t,states)
        print(f'Anhatmonic Osillator - Amplitude{A}:Period = {period:.4f}')
        plot_time_evolution(t,states,f'Time Evolution of Anharmonic Osilator(A={A})')
        
    # TODO: 任务4 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 比较简谐和非谐振子
    initial_state = np.array([1.0, 0.0])
    t, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, 'Phase Space Trajectory of Harmonic Oscillator')

    t, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, 'Phase Space Traject


if __name__ == "__main__":
    main()
