# 靡不有初，鲜克有终
# 开发时间：2022/10/25 18:02
"""
演示共轭梯度法解决一个凸函数的最优化问题的例子
优化函数：凸二次函数，n=2
        f(x_1,x_2)=(x_1)**2+2(x_2)**2+2(x_1)*(x_2)-5
"""

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import eig

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
vector_x = np.array([[x1], [x2]])  # 定义二维变量（向量）
mtx_Q = np.matrix([[2, 2], [2, 4]])  # 目标函数的二次型矩阵
f = np.array(vector_x.T*(0.5*mtx_Q)*vector_x)[0][0]  # 目标函数 xT Q x
gx1 = sp.diff(f, x1)
gx2 = sp.diff(f, x2)
gx1x1 = sp.diff(gx1, x1)
gx1x2 = sp.diff(gx1, x2)
gx2x2 = sp.diff(gx2, x2)
gx2x1 = sp.diff(gx2, x1)


def function_value(vector):  # 求函数值
    f_value = f.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    return f_value


def gradient(vector):  # 求梯度向量及模长
    g_x1 = gx1.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    g_x2 = gx2.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    vector_g = np.array([[g_x1], [g_x2]])
    modulus = (pow(g_x1, 2) + pow(g_x2, 2)) ** 0.5
    return [vector_g, modulus]


def hesse(vector):  # 求海塞阵修正后的数值,矩阵形式给出
    gx1x1_value = float(gx1x1.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx1x2_value = float(gx1x2.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx2x1_value = float(gx2x1.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx2x2_value = float(gx2x2.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    mtx_H_value = np.matrix([[gx1x1_value, gx1x2_value], [gx2x1_value, gx2x2_value]])
    return mtx_H_value


def generate_grid(min_x_1, max_x_1, min_x_2, max_x_2, delta):  # 三维网格底图，参数含范围和颗粒度
    x_1_array = np.arange(min_x_1, max_x_1, delta)
    x_2_array = np.arange(min_x_2, max_x_2, delta)
    x1_array, x2_array = np.meshgrid(x_1_array, x_2_array)
    f_value_array = pow(x1_array, 2)+2*pow(x2_array, 2)+2*x1_array*x2_array-5
    return x1_array, x2_array, f_value_array


def conjugate_gradient_method(vector_initial, epsilon):  # 共轭梯度法
    k = 0
    vector_x_pre = vector_initial
    g_pre = gradient(vector_x_pre)[0]  # 初始梯度
    dir_pre = -g_pre  # 初始方向（负梯度）
    x_1_lst = [vector_x_pre[0][0]]
    x_2_lst = [vector_x_pre[1][0]]
    f_value_lst = [function_value(vector_x_pre)]
    while float(gradient(vector_x_pre)[1]) > epsilon and k <= 300:
        step_pre = -(np.dot(gradient(vector_x_pre)[0].T, dir_pre)).item()/(dir_pre.T*mtx_Q*dir_pre).item()  # 初始步长
        vector_x_new = np.array(vector_x_pre + step_pre * dir_pre)
        beta = (np.dot(gradient(vector_x_new)[0].T, np.dot(mtx_Q, dir_pre))).item()/(np.dot(dir_pre.T,np.dot(mtx_Q,dir_pre))).item()
        dir_new = -gradient(vector_x_new)[0] + beta * dir_pre

        x_1_lst.append(vector_x_new[0][0])
        x_2_lst.append(vector_x_new[1][0])
        f_value_lst.append(function_value(vector_x_new))

        vector_x_pre = vector_x_new
        dir_pre = dir_new
        k = k + 1

    f_value = function_value(vector_x_pre)
    return vector_x_pre, f_value, k, x_1_lst, x_2_lst, f_value_lst


def plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst):  # 绘制二维图像并保存
    plt.figure()
    plt.contourf(X1, X2, F, 10)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x_1_lst, x_2_lst, c='r', linewidth=1.5)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//10.线性共轭梯度法2D.png')
    plt.show()


def plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst,f_value_lst):  # 绘制三维图像并保存
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X1, X2, F, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x_1_lst, x_2_lst, f_value_lst, c='r', linewidth=1.5)
    plt.colorbar(p, shrink=0.8)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//10.线性共轭梯度法法3D.png')
    plt.show()


if __name__ == "__main__":
    vector = np.array([[-5], [7]])
    vector_x_opt, value, times, x_1_lst, x_2_lst, f_value_lst = conjugate_gradient_method(vector, 0.001)
    X1, X2, F = generate_grid(-10, 10, -10, 10, 0.05)  # 生成用于三维展示的函数面
    print("最优点坐标", vector_x_opt[0][0], ',', vector_x_opt[1][0])
    print("最优值", value)
    print('迭代次数', times)
    plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst)
    plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst, f_value_lst)