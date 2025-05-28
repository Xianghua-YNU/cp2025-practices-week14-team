import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    # TODO: 在此实现受迫单摆的ODE方程
    theta,omega =state
    dtheta_dt = omega  #角速度
    domega_dt = -(g/l)*np.sin(theta)+(C/l)*np.cos(theta)*np.sin(Omega*t)
    return [dtheta_dt,domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    # TODO: 使用solve_ivp求解受迫单摆方程
    # 提示: 需要调用forced_pendulum_ode函数
    def ode_func(t,state):
        return forced_pendulum_ode(t,state,l,g,C,Omega)

    #调用solve_ivp求解
    sol = solve_ivp(y:ode_func,
                    t_span,y0,
                    t_eval=np.linspace(t_span[0],t_span[1],2000),  #生成2000个时间点
                    method='Radau',  #适用于刚性问题
                    rtol=1e-6
                    atol=1e-9#控制精度
                 )
    return sol.t,sol.y[0]  #返回时间和角度


def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    # TODO: 实现共振频率查找功能
    # 提示: 需要调用solve_pendulum函数并分析结果
    if Omega_range is None:
        Omage0=np.sqrt(g/l)  #小角度近似下的自然频率
        Omega_range=np.linspa(Omega0/2,2*Omega0,50)  #默认扫描范围

    amplitudes=[]
    for Omega in Omega_range:
        #求解单摆运动
        t,theta =solve_pendulum(1,g,C,Omega,t_span,y0)
        #计算稳态振幅（取后1/2数据）
        steady_start=int(0.66*len(theta))
        steady_theta=theta[steady_start:]
        #计算峰峰值并取半作为振幅
        amplitude = (np.max(steady_theta)-np.min(steady_theta))/2
        amplitudes.append(amplitude)
    return Omega_range,np.array(amplitudes)


def plot_results(t, theta, title):
    """绘制结果"""
    # 此函数已提供完整实现，学生不需要修改
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    # TODO: 调用solve_pendulum和plot_results
    t,theta = solve_pendulum()
    plot_results(t,theta,"Forced Pendulum (Omega=5)")

    # 任务2: 探究共振现象
    # TODO: 调用find_resonance并绘制共振曲线
    Omega_range, amplitudes = find_resonance()

    # 找到共振频率并绘制共振情况
    # TODO: 实现共振频率查找和绘图
    #绘制共振曲线
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range,amplitudes,'b-',label='Amplitudes')
    #标注固有频率（小角度近似）
    omega_natural=np.sqrt(9.81/0.1)
    plt.axvline(omega_natural,color='r',linestyle='--',label=f'Natural Frequency:{omega_natural:.2f}rad/s')
    plt.xlabel('Drive Frequency(rad/s)')
    plt.ylabel('Amplitude(rad)')
    plt.title('Resonance Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    #找到共振频率并绘制其时间序列
    resonance_Omega=Omega_range[np.argmax(amplitudes)]
    t_res,theta_res=solve_pendulum(Omega=resonance_Omega)
    print(f'Resonance frequency:({t_res:.2f}rad/s')
    plot_results(t_res,theta_res,f"Resonance at Omega={resonance_Omega:.2f}rad/s")


if __name__ == '__main__':
    main()
