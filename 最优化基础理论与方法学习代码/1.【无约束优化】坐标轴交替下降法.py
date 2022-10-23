# 靡不有初，鲜克有终
# 开发时间：2022/10/20 19:38
"""
演示坐标轴交替下降法解决一个凸函数的最优化问题的例子
优化函数：
        f(x_1,x_2)=(x_1)**2+2(x_2)**2+2(x_1)*(x_2)-5
"""
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
f = pow(x1, 2) + 2 * pow(x2, 2) + 2 * x1 * x2 - 5
gx1 = sp.diff(f, x1)  # x1偏导
gx2 = sp.diff(f, x2)  # x2偏导


def function_value(x_1, x_2):  # 求函数值
    f_value = f.evalf(subs={x1: x_1, x2: x_2})
    return f_value


def gradient_modulus(x_1, x_2):  # 求梯度
    g_x1 = gx1.evalf(subs={x1: x_1, x2: x_2})
    g_x2 = gx2.evalf(subs={x1: x_1, x2: x_2})
    return (pow(g_x1, 2) + pow(g_x2, 2))**0.5


def generate_grid(min_x_1, max_x_1, min_x_2, max_x_2, delta):  # 三维网格底图，参数含范围和颗粒度
    x_1_array = np.arange(min_x_1, max_x_1, delta)
    x_2_array = np.arange(min_x_2, max_x_2, delta)
    x1_array, x2_array = np.meshgrid(x_1_array, x_2_array)
    f_value_array = pow(x1_array, 2)+2*pow(x2_array, 2)+2*x1_array*x2_array-5
    return x1_array, x2_array, f_value_array


def step_length(x_1, x_2, search_dir):  # 线搜索确定步长
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    if search_dir == 1:  # 搜索方向x_1轴
        x_1_function = sp.solve(gx1, x1)[0]
        step = x_1_function.evalf(subs={x2: x_2}) - x_1
        return step
    elif search_dir == 2:  # 搜索方向x_2轴
        x_2_function = sp.solve(gx2, x2)[0]
        step = x_2_function.evalf(subs={x1: x_1}) - x_2
        return step


def coordinate_descent_method(start_x_1, start_x_2, epsilon):  # 交替坐标轴下降算法
    k = 0
    x_1 = start_x_1
    x_2 = start_x_2
    x_1_lst = [start_x_1]
    x_2_lst = [start_x_2]
    f_value_lst = [function_value(x_1, x_2)]
    while gradient_modulus(x_1, x_2) > epsilon and k <= 100:
        search_dir = 1
        step_x_1 = step_length(x_1, x_2, search_dir)
        x_1 = x_1 + step_x_1
        x_1_lst.append(x_1)
        x_2_lst.append(x_2)
        f_value_lst.append(function_value(x_1, x_2))

        search_dir = 2
        step_x_2 = step_length(x_1, x_2, search_dir)
        x_2 = x_2 + step_x_2
        x_1_lst.append(x_1)
        x_2_lst.append(x_2)
        f_value_lst.append(function_value(x_1, x_2))

        k = k + 1
    f_value = function_value(x_1, x_2)
    return x_1, x_2, f_value, k, x_1_lst, x_2_lst, f_value_lst


def plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst):  # 绘制二维图像并保存
    plt.figure()
    plt.contourf(X1, X2, F, 10)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x_1_lst, x_2_lst, c='r', linewidth=1.5)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//1.坐标轴交替下降法2D.png')
    plt.show()


def plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst,f_value_lst):  # 绘制三维图像并保存
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X1, X2, F, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x_1_lst, x_2_lst, f_value_lst, c='r', linewidth=1.5)
    plt.colorbar(p, shrink=0.8)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//1.坐标轴交替下降法3D.png')
    plt.show()


if __name__ == "__main__":
    x1_opt, x2_opt, value, times, x_1_lst, x_2_lst, f_value_lst = coordinate_descent_method(-5, 7, 0.01)
    X1, X2, F = generate_grid(-10, 10, -10, 10, 0.05)  # 生成用于三维展示的函数面
    print("最优点坐标", x1_opt, ',', x2_opt)
    print("最优值", value)
    print('迭代次数', times)
    plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst)
    plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst, f_value_lst)

