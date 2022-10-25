# 靡不有初，鲜克有终
# 开发时间：2022/10/24 11:23
"""
演示牛顿法解决一个凸函数的最优化问题的例子
优化函数：凸二次函数，n=2
        f(x_1,x_2)=(x_1)**2+2(x_2)**2+2(x_1)*(x_2)-5
"""
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
vector_x = np.array([[x1], [x2]])  # 定义二维变量（向量）
mtx_A = np.matrix([[2, 2], [2, 4]])  # 目标函数的二次型矩阵
f = np.array(vector_x.T*(0.5*mtx_A)*vector_x)[0][0]  # 目标函数
print(f)
print(type(f))
gx1 = sp.diff(f, x1)
gx2 = sp.diff(f, x2)
gx1x1 = sp.diff(gx1, x1)
gx1x2 = sp.diff(gx1, x2)
gx2x2 = sp.diff(gx2, x2)
gx2x1 = sp.diff(gx2, x1)
mtx_H = np.matrix([[gx1x1, gx1x2], [gx2x1, gx2x2]])  # Hesse阵
mtx_H = mtx_H.astype(np.float)  # Hesse阵,类型转换用于求逆


def function_value(vector):  # 求函数值
    f_value = f.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    return f_value


def gradient(vector):  # 求梯度向量及模长
    g_x1 = gx1.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    g_x2 = gx2.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    vector_g = np.array([[g_x1], [g_x2]])
    modulus = (pow(g_x1, 2) + pow(g_x2, 2)) ** 0.5
    return [vector_g, modulus]


def generate_grid(min_x_1, max_x_1, min_x_2, max_x_2, delta):  # 三维网格底图，参数含范围和颗粒度
    x_1_array = np.arange(min_x_1, max_x_1, delta)
    x_2_array = np.arange(min_x_2, max_x_2, delta)
    x1_array, x2_array = np.meshgrid(x_1_array, x_2_array)
    f_value_array = pow(x1_array, 2)+2*pow(x2_array, 2)+2*x1_array*x2_array-5
    return x1_array, x2_array, f_value_array


def newton_method(vector_initial, epsilon):  # 牛顿法
    k = 0
    vector_x_pre = vector_initial
    x_1_lst = [vector_x_pre[0][0]]
    x_2_lst = [vector_x_pre[1][0]]
    f_value_lst = [function_value(vector_x_pre)]
    while gradient(vector_x_pre)[1] > epsilon and k <= 100:
        vector_x_new = np.array(vector_x_pre - np.linalg.inv(mtx_H)*gradient(vector_x_pre)[0])
        vector_x_new.astype(np.float)
        x_1_lst.append(vector_x_new[0][0])
        x_2_lst.append(vector_x_new[1][0])
        f_value_lst.append(function_value(vector_x_new))
        k = k + 1
        vector_x_pre = vector_x_new
    f_value = function_value(vector_x_pre)
    return vector_x_pre, f_value, k, x_1_lst, x_2_lst, f_value_lst


def plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst):  # 绘制二维图像并保存
    plt.figure()
    plt.contourf(X1, X2, F, 10)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x_1_lst, x_2_lst, c='r', linewidth=1.5)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//4.牛顿法2D.png')
    plt.show()


def plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst,f_value_lst):  # 绘制三维图像并保存
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X1, X2, F, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x_1_lst, x_2_lst, f_value_lst, c='r', linewidth=1.5)
    plt.colorbar(p, shrink=0.8)
    plt.savefig('C://Users//张晨皓//Desktop//最优化基础理论与方法学习代码//图片//4.牛顿法3D.png')
    plt.show()


if __name__ == "__main__":
    vector = np.array([[-5], [7]])
    vector_x_opt, value, times, x_1_lst, x_2_lst, f_value_lst = newton_method(vector, 0.01)
    X1, X2, F = generate_grid(-10, 10, -10, 10, 0.05)  # 生成用于三维展示的函数面
    print("最优点坐标", vector_x_opt[0][0], ',', vector_x_opt[1][0])
    print("最优值", value)
    print('迭代次数', times)
    plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst)
    plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst, f_value_lst)



