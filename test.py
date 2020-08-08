import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


# 解析関数の定義
def f(X):
    return X * np.sin(X)

# データセットの作成
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X)

# データセットの可視化
plot_X = np.atleast_2d(np.linspace(0, 10, 1000)).T
fig = plt.figure(figsize=(8, 6))
plt.plot(plot_X, f(plot_X), 'k')
plt.plot(X, y, 'r.', markersize=16)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.ylim(-8, 12)
plt.legend(['$y = x*\sin(x)$', 'Observed values'], loc='upper left', fontsize=16)
plt.tick_params(labelsize=16)
# plt.show()


# データのscaling
# scikit-learnに実装されているStandardScalerを利用
# 説明変数のscalingはしなくても問題ありませんが、目的変数のscalingは必須です(平均の事前分布を与えるのが難しいため)
scaler_y = StandardScaler().fit(y) 

# GPモデルの構築
kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 
gpr.fit(X, scaler_y.transform(y))

# kernel関数のパラメータの確認
# print(gpr.kernel_)
# 1.36**2 * RBF(length_scale=3.07) + WhiteKernel(noise_level=0.061)

# plot_Xに対する平均および標準偏差の予測
pred_mu, pred_sigma = gpr.predict(plot_X, return_std=True)
pred_mu = scaler_y.inverse_transform(pred_mu)
pred_sigma = pred_sigma.reshape(-1, 1) * scaler_y.scale_

# 各xに対する95%信頼区間を表示
fig = plt.figure(figsize=(8, 6))
plt.plot(plot_X, f(plot_X), 'k')
plt.plot(X, y, 'r.', markersize=16)
plt.plot(plot_X, pred_mu, 'b')
# データが正規分布に従う場合、95%信頼区間は平均から標準偏差の1.96倍以内の区間となる
plt.fill_between(plot_X.squeeze(), (pred_mu - 1.9600 * pred_sigma).squeeze(), (pred_mu + 1.9600 * pred_sigma).squeeze())
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.ylim(-8, 12)
plt.legend(['$y = x*\sin(x)$', 'Observed values', 'Predicted mean', '95% confidence interval'], loc='upper left', fontsize=16)
plt.tick_params(labelsize=16)
# plt.show()