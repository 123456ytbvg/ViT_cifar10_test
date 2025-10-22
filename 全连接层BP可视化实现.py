import numpy as np
import matplotlib.pyplot as plt


N = 233

s = np.random.rand(4) - 0.5
s[3] *= 3
points = np.random.rand(N,4) - 0.5
points[:,:3] *= 10
points[:,3] = ((np.sign(np.sum(s[:3]*points[:,:3],axis=1) + s[3])*0.5)+0.5).astype(int)

class FC():
    def __init__(self):
        self.W = np.random.randn(4) - 0.5
        self.lr = 10
    def predict(self,x):
        z = np.dot(x,self.W[:3])+self.W[3]
        p = 1 / (1 + np.exp(-z/4))
        output = 1 if p > 0.5 else 0
        return output
    def train(self,x,label):
        z = np.dot(x,self.W[0:3]) + self.W[3]
        p = 1/(1+np.exp(-z/4))
        bp1 = -(label - p)
        loss = np.abs(bp1)
        self.W[:3] -= self.lr *bp1*x
        self.W[3] -= self.lr *bp1
        return loss

model = FC()
Epoch = 233
for i in range(Epoch):
    loss = 0
    for j in range(N):

        x,y = points[j,:3],points[j,3]
        loss += model.train(x,y)
    loss /= N
    print(f"轮次{i}损失：{loss}")

print(f"模型参数：{model.W},\n,数据集生成依据{s}")

output = np.zeros(N)
for i in range(N):
    output[i] = model.predict(points[i][0:3])
error = np.sum(np.abs(output-points[:,3]))/N
print(f"预测错误率：{error}")

######################################################
import plotly.graph_objects as go
import numpy as np

# 分离两类点
a_points = points[points[:, 3] == 1][:, :3]
b_points = points[points[:, 3] == 0][:, :3]

# 创建散点图
scatter_a = go.Scatter3d(
    x=a_points[:, 0], y=a_points[:, 1], z=a_points[:, 2],
    mode='markers',
    marker=dict(size=4, color='red', opacity=0.8),
    name='Class A'
)

scatter_b = go.Scatter3d(
    x=b_points[:, 0], y=b_points[:, 1], z=b_points[:, 2],
    mode='markers',
    marker=dict(size=4, color='blue', opacity=0.8),
    name='Class B'
)

# 获取参数
learned_w1, learned_w2, learned_w3, learned_b = model.W
true_w1, true_w2, true_w3, true_b = s

# 计算所有点的最大范围，确保三个轴用同样的尺度
all_points = np.vstack([a_points, b_points])
max_range = max(
    all_points[:, 0].max() - all_points[:, 0].min(),
    all_points[:, 1].max() - all_points[:, 1].min(),
    all_points[:, 2].max() - all_points[:, 2].min()
) / 2.0

mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

# 创建网格 - 使用统一的范围
x_range = np.linspace(mid_x - max_range, mid_x + max_range, 15)
y_range = np.linspace(mid_y - max_range, mid_y + max_range, 15)
xx, yy = np.meshgrid(x_range, y_range)

# 计算平面
zz_learned = (-learned_w1 * xx - learned_w2 * yy - learned_b) / learned_w3
zz_true = (-true_w1 * xx - true_w2 * yy - true_b) / true_w3

# 创建平面
plane_learned = go.Surface(
    x=xx, y=yy, z=zz_learned,
    colorscale='Greens',
    opacity=0.6,
    showscale=False,
    name='Learned Plane'
)

plane_true = go.Surface(
    x=xx, y=yy, z=zz_true,
    colorscale='Oranges',
    opacity=0.6,
    showscale=False,
    name='True Plane'
)

# 创建图形
fig = go.Figure(data=[scatter_a, scatter_b, plane_learned, plane_true])

# 关键设置：aspectmode='cube' 确保三个轴比例一致
fig.update_layout(
    title='3D Visualization - Equal Axis Scaling',
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Feature 3',
        # 关键设置：立方体比例模式
        aspectmode='cube',
        # 设置统一的坐标轴范围
        xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
        yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
        zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=800,
    height=600
)

fig.show()