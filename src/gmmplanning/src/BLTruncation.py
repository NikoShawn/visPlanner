import numpy as np
from scipy.stats import norm as scipy_norm  # Rename to avoid conflicts
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm

# Function to calculate the multivariate normal PDF
def multivariate_normal(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized way
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N

# Barber and Lopez-Perez Truncation algorithm
def BLTruncation(mu_0, Sigma_0, a, b):
    # 确保所有输入都是正确的numpy数组
    mu_0 = np.asarray(mu_0)
    Sigma_0 = np.asarray(Sigma_0)
    a = np.asarray(a).reshape(-1, 1)  # 转换为列向量 (n,1)
    
    # Reverse direction of the normal vector
    a = -a
    b = -b
    
    # Calculate key parameters
    q0 = a.T @ Sigma_0 @ a  # 结果应该是标量
    alpha = (b - a.T @ mu_0) / np.sqrt(q0)
    
    # Calculate truncation parameter
    lambda_val = scipy_norm.pdf(alpha) / scipy_norm.cdf(alpha)  # Use scipy_norm instead of norm
    
    # Handle numerical issues
    if np.isnan(lambda_val):
        lambda_val = -alpha
    
    # Update the distribution parameters
    mu1 = a.T @ mu_0 - lambda_val * np.sqrt(q0)
    var1 = q0 * (1 - lambda_val**2 - alpha * lambda_val)
    
    # Calculate mean shift
    dmu = (Sigma_0 @ a) * (a.T @ mu_0 - mu1) / q0
    
    # Update mean and covariance
    mu = mu_0 - dmu.flatten()  # 确保结果是1D向量
    
    # 修正矩阵计算部分
    term = (Sigma_0 @ a) @ (a.T @ Sigma_0)  # 显式矩阵乘法
    Sigma = Sigma_0 - term * (q0 - var1) / (q0**2)
    
    return mu, Sigma

# Apply individual truncations
def IndividualTruncations(mu_0, Sigma_0, A, b):
    # 确保输入是numpy数组
    mu_0 = np.asarray(mu_0)
    Sigma_0 = np.asarray(Sigma_0)
    A = np.asarray(A)
    b = np.asarray(b)
    
    # 存储每条截断线独立截断后的结果
    truncated_distributions = []
    
    # 将原始分布也加入结果列表
    truncated_distributions.append((mu_0, Sigma_0, None, None, "Original"))
    
    # 对每条截断线分别进行截断
    for i in range(len(b)):
        a_i = A[i]
        b_i = b[i]
        mu_i, Sigma_i = BLTruncation(mu_0, Sigma_0, a_i, b_i)  # 从原始分布开始截断
        truncated_distributions.append((mu_i, Sigma_i, a_i, b_i, f"Boundary {i+1}"))
    
    return truncated_distributions

# Visualize individual truncations
def visualize_individual_truncations(mu_0, Sigma_0, A, b):
    # 获取所有独立截断的结果
    truncated_distributions = IndividualTruncations(mu_0, Sigma_0, A, b)
    
    # Create a grid for evaluation
    x = np.linspace(mu_0[0] - 3*np.sqrt(Sigma_0[0,0]), mu_0[0] + 3*np.sqrt(Sigma_0[0,0]), 100)
    y = np.linspace(mu_0[1] - 3*np.sqrt(Sigma_0[1,1]), mu_0[1] + 3*np.sqrt(Sigma_0[1,1]), 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Create figure with white background
    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.subplot(111)
    ax.set_facecolor('#f8f8f8')
    
    # 定义不同分布的颜色
    n_dist = len(truncated_distributions)
    # 使用不同的颜色映射来区分原始分布和截断分布
    colors = ['navy'] + list(plt.cm.tab10(np.linspace(0, 1, n_dist-1)))
    
    # 线段颜色
    line_colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange', 'purple', 'brown']
    
    # 绘制原始分布
    original_mu, original_Sigma = truncated_distributions[0][0], truncated_distributions[0][1]
    original_pdf = multivariate_normal(pos, original_mu, original_Sigma)
    
    # 原始分布使用透明的灰色填充
    cs_original = ax.contour(X, Y, original_pdf, levels=15, 
                            colors='navy', alpha=0.4, linestyles='--', linewidths=1.0)
    ax.scatter(original_mu[0], original_mu[1], color='navy', edgecolor='black',
              s=100, marker='o', label='Original Mean', zorder=10)
    
    # 绘制每个截断线及其对应的截断分布
    for i in range(1, n_dist):
        mu, Sigma, a, b_val, label = truncated_distributions[i]
        
        # 颜色选择
        color = line_colors[(i-1) % len(line_colors)]
        line_label = f"Boundary {i}"
        
        # 绘制截断线
        x_vals = np.array([x[0], x[-1]])
        if a[1] != 0:
            y_vals = (b_val - a[0] * x_vals) / a[1]
            valid_indices = (y_vals >= y[0]) & (y_vals <= y[-1])
            x_vals = x_vals[valid_indices]
            y_vals = y_vals[valid_indices]
        else:
            # 垂直线
            x_val = b_val / a[0]
            x_vals = np.array([x_val, x_val])
            y_vals = np.array([y[0], y[-1]])
        
        # 绘制截断线
        ax.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=2.5, 
                label=line_label, zorder=5)
        
        # 计算并绘制截断后的分布
        pdf = multivariate_normal(pos, mu, Sigma)
        
        # 使用与截断线相近的颜色，但更浅一些
        lighter_color = list(plt.cm.get_cmap('Pastel1')(i-1))
        
        # 绘制填充等高线
        cs = ax.contourf(X, Y, pdf, levels=15, 
                         colors=[lighter_color], alpha=0.4)
        
        # 绘制轮廓线
        cs_lines = ax.contour(X, Y, pdf, levels=5, 
                             colors=[color], alpha=0.7, linewidths=1.0)
        
        # 绘制均值点
        ax.scatter(mu[0], mu[1], color=color, edgecolor='black',
                  s=80, marker='*', label=f"Mean after {line_label}", zorder=10)
        
        # 绘制有效区域
        # 创建遮罩来表示有效区域
        valid_mask = np.ones_like(X, dtype=bool)
        for j in range(len(X)):
            for k in range(len(X[0])):
                point = np.array([X[j, k], Y[j, k]])
                if np.dot(a, point) > b_val:  # 如果点在无效区域
                    valid_mask[j, k] = False
        
        # 显示有效区域边界
        ax.contour(X, Y, valid_mask.astype(float), levels=[0.5], 
                  colors=[color], linestyles='dotted', linewidths=1.0, alpha=0.5)
    
    # 添加标签和图例
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Multiple Gaussian Truncations', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 创建分组图例
    handles, labels = ax.get_legend_handles_labels()
    
    # 创建自定义图例，分组显示
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return plt.gcf()

# Main execution
if __name__ == "__main__":
    # Original Gaussian distribution parameters
    mu_0 = np.array([-12.08, -8.18])  # Mean
    Sigma_0 = np.array([[0.80, 0.0],   # Covariance matrix (assuming diagonal)
                        [0.0, 0.80]])

    # Extract a, b, c coefficients from line equations
    lines = [
        (-0.6866, 0.7270, -3.9360),   # Line 1
        (-0.1902, 0.9817, 3.6661),    # Line 2
        (0.8798, -0.4754, 5.4446),    # Line 3
        (-0.6385, -0.7696, -15.1664)  # Line 4
    ]

    # Convert to normal vectors and position parameters
    A = []  # Will contain the normal vectors
    b = []  # Will contain the position parameters

    for a_coef, b_coef, c_coef in lines:
        # Create normal vector
        normal = np.array([a_coef, b_coef])
        
        # Normalize
        norm = np.linalg.norm(normal)
        normal = normal / norm
        c_normalized = c_coef / norm
        
        # Add to lists
        A.append(normal)
        b.append(-c_normalized)  # Note the negative sign to convert from c to b

    # Convert lists to numpy arrays
    A = np.array(A)
    b = np.array(b)

    # Print the original and transformed parameters
    print("Original Gaussian Distribution:")
    print(f"Mean: {mu_0}")
    print(f"Covariance:\n{Sigma_0}")
    
    print("\nTruncation Lines (ax + by + c = 0):")
    for i, (a_coef, b_coef, c_coef) in enumerate(lines):
        print(f"Line {i+1}: {a_coef:.4f}x + {b_coef:.4f}y + {c_coef:.4f} = 0")
    
    print("\nTransformed Parameters for BLTruncation (a·x ≤ b):")
    for i, (a_vec, b_val) in enumerate(zip(A, b)):
        print(f"Line {i+1}: normal vector={a_vec}, position={b_val:.4f}")

    # Visualize the truncations
    fig = visualize_individual_truncations(mu_0, Sigma_0, A, b)
    plt.savefig('gaussian_truncations.png', dpi=300, bbox_inches='tight')
    plt.show()