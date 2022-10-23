# 靡不有初，鲜克有终
# 开发时间：2022/10/23 19:30
"""
演示最速下降法解决一个凸函数的最优化问题的例子
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
print(gx1)
print(gx2)


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


def line_search(x_1, x_2):  # 利用梯度对应的直线方程，用于线搜索,得到下一个最优的搜索点
    g_x1 = gx1.evalf(subs={x1: x_1, x2: x_2})
    g_x2 = gx2.evalf(subs={x1: x_1, x2: x_2})
    line_function = (x1-x_1)/g_x1-(x2-x_2)/g_x2
    x2_line_function = sp.solve(line_function, x2)[0]  # 用x1来表示x2
    search_function = pow(x1, 2) + 2 * pow(x2_line_function, 2) + 2 * x1 * x2_line_function - 5  # 得到关于x1的公式
    gradient_search_function = sp.diff(search_function, x1)  # 关于x1的导数公式
    x1_opt = sp.solve(gradient_search_function, x1)[0]  # 求导（精确线搜索）得到x1为当前方向最优解
    x2_opt = x2_line_function.evalf(subs={x1: x1_opt})
    return x1_opt, x2_opt


def gradient_descent_method(start_x_1, start_x_2, epsilon):  # 最速下降算法
    k = 0
    x_1 = start_x_1
    x_2 = start_x_2
    x_1_lst = [start_x_1]
    x_2_lst = [start_x_2]
    f_value_lst = [function_value(x_1, x_2)]
    while gradient_modulus(x_1, x_2) > epsilon and k <= 100:
        x_1, x_2 = line_search(x_1, x_2)
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
    plt.savefig('3.最速下降法2D.png')
    plt.show()


def plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst,f_value_lst):  # 绘制三维图像并保存
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X1, X2, F, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x_1_lst, x_2_lst, f_value_lst, c='r', linewidth=1.5)
    plt.colorbar(p, shrink=0.8)
    plt.savefig('3.最速下降法3D.png')
    plt.show()


if __name__ == "__main__":
    x1_opt, x2_opt, value, times, x_1_lst, x_2_lst, f_value_lst = gradient_descent_method(-5, 7, 0.01)
    X1, X2, F = generate_grid(-10, 10, -10, 10, 0.05)  # 生成用于三维展示的函数面
    print("最优点坐标", x1_opt, ',', x2_opt)
    print("最优值", value)
    print('迭代次数', times)
    plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst)
    plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst, f_value_lst)
