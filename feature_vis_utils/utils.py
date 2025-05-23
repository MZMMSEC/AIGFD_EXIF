import pdb

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

def tsne_plot_save_dir(features, labels, result_dir, savename='tsne_visualization.png'):
    # 创建 t-SNE 模型并进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 获取唯一的标签类别
    unique_labels = np.unique(labels)

    # 定义颜色映射
    # colors = ['red', 'blue', 'green']
    colors = ['#EC7677', '#7EC6F7']

    # 绘制可视化图形
    plt.figure(figsize=(8, 8))
    for label in unique_labels:
        # 获取属于当前类别的特征和对应的 t-SNE 降维结果
        label_features = features_tsne[labels == label]

        # 获取当前类别的颜色
        color = colors[label]

        # 绘制当前类别的散点图
        plt.scatter(label_features[:, 0], label_features[:, 1], color=color, label=f'Label {label}')

    # 添加图例
    # plt.legend()

    # 添加标题和标签
    plt.title('t-SNE Visualization of Features')
    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)

    # 显示图形
    # 保存图像
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, savename)
    plt.savefig(save_path)
    # plt.show()
