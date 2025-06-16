import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors
def paint(data,lower,upper,func_num,iter):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    # 创建一个200x3的随机数据集作为示例
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter')
    ax.set_xlim(lower,upper)  # 设置x轴范围为0到1
    ax.set_ylim(lower,upper)  # 设置y轴范围为0到1
    ax.set_zlim(lower,upper)  # 设置z轴范围为0到1
    plt.savefig('E:/tcde/F'+str(func_num)+'/'+str(iter)+'allpopulation.png', dpi=300, bbox_inches='tight')
    # 保存图形为PNG文件
    plt.close('all')

def plot_3d_circles_with_random_colors(circle_data,func_num,iter,lower,upper,NR):
    """

    """
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_names = list(mcolors.CSS4_COLORS.keys())
    # 绘制每个圆
    for i,circle in enumerate(circle_data):
        x, y, z, r = circle
        color = color_names[i % len(color_names)] # 生成随机颜色，RGB值
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_circle = r * np.outer(np.cos(u), np.sin(v)) + x
        y_circle = r * np.outer(np.sin(u), np.sin(v)) + y
        z_circle = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x_circle, y_circle, z_circle, color=color, alpha=0.5)
    ax.set_xlim(lower,upper)  # 设置x轴范围为0到1
    ax.set_ylim(lower,upper)  # 设置y轴范围为0到1
    ax.set_zlim(lower,upper)  # 设置z轴范围为0到1
    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D-TS')
    plt.savefig('E:/tcde/F' + str(func_num) + '/NR_' + str(NR) + 'iter_'+ str(iter) +'len_'+ str(len(circle_data))+'TS.png', dpi=300, bbox_inches='tight')
    plt.close('all')
