import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm

def BLTruncation(mu_0, Sigma_0, a, b):
    """
    Barrier Likelihood Truncation method for Gaussian distributions
    
    Parameters:
    -----------
    mu_0 : numpy.ndarray
        Original mean vector (shape (n,))
    Sigma_0 : numpy.ndarray
        Original covariance matrix (shape (n,n))
    a : numpy.ndarray
        Normal vector of the truncation plane (shape (n,))
    b : float
        Position parameter of the truncation plane
    
    Returns:
    --------
    mu : numpy.ndarray
        Truncated mean vector (shape (n,))
    Sigma : numpy.ndarray
        Truncated covariance matrix (shape (n,n))
    """
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
    lambda_val = norm.pdf(alpha) / norm.cdf(alpha)
    
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

# Function to calculate the multivariate normal PDF
def multivariate_normal(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized way
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N

def visualize_truncation(mu_0, Sigma_0, mu_trunc, Sigma_trunc, a, b):
    """
    Visualize the original and truncated Gaussian distributions.
    
    Parameters:
    -----------
    mu_0 : numpy.ndarray
        Original mean vector (shape (2,))
    Sigma_0 : numpy.ndarray
        Original covariance matrix (shape (2,2))
    mu_trunc : numpy.ndarray
        Truncated mean vector (shape (2,))
    Sigma_trunc : numpy.ndarray
        Truncated covariance matrix (shape (2,2))
    a : numpy.ndarray
        Normal vector of the truncation plane (shape (2,))
    b : float
        Position parameter of the truncation plane
    """
    # Create a grid for evaluation
    x = np.linspace(mu_0[0] - 3*np.sqrt(Sigma_0[0,0]), mu_0[0] + 3*np.sqrt(Sigma_0[0,0]), 100)
    y = np.linspace(mu_0[1] - 3*np.sqrt(Sigma_0[1,1]), mu_0[1] + 3*np.sqrt(Sigma_0[1,1]), 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDFs
    Z_original = multivariate_normal(pos, mu_0, Sigma_0)
    Z_truncated = multivariate_normal(pos, mu_trunc, Sigma_trunc)
    
    # Create figure with white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    fig.set_facecolor('white')
    
    # Plot original Gaussian with improved color scheme
    levels = 15  # More contour levels for smoother visualization
    cs1 = ax1.contourf(X, Y, Z_original, levels=levels, cmap=cm.Blues, alpha=0.9)
    ax1.contour(X, Y, Z_original, levels=levels, colors='navy', alpha=0.5, linewidths=0.5)
    ax1.scatter(mu_0[0], mu_0[1], c='red', marker='x', s=100, label='Mean', zorder=5)
    ax1.set_title('Original Gaussian Distribution', fontsize=14, fontweight='bold')
    ax1.set_facecolor('#f8f8f8')  # Light gray background
    
    # Plot truncated Gaussian with improved color scheme
    cs2 = ax2.contourf(X, Y, Z_truncated, levels=levels, cmap=cm.Blues, alpha=0.9)
    ax2.contour(X, Y, Z_truncated, levels=levels, colors='navy', alpha=0.5, linewidths=0.5)
    ax2.scatter(mu_trunc[0], mu_trunc[1], c='red', marker='x', s=100, label='Mean', zorder=5)
    ax2.set_title('Truncated Gaussian Distribution', fontsize=14, fontweight='bold')
    ax2.set_facecolor('#f8f8f8')  # Light gray background
    
    # Draw the truncation line
    # Find points to represent the boundary
    x_vals = np.array([x[0], x[-1]])
    # For the line ax + by + c = 0, where a is a[0], b is a[1], and c is -b
    if a[1] != 0:
        y_vals = (b - a[0] * x_vals) / a[1]
        valid_indices = (y_vals >= y[0]) & (y_vals <= y[-1])
        x_vals = x_vals[valid_indices]
        y_vals = y_vals[valid_indices]
    else:
        # Vertical line
        x_val = b / a[0]
        x_vals = np.array([x_val, x_val])
        y_vals = np.array([y[0], y[-1]])
    
    # Plot truncation line on both subplots with improved visibility
    for ax in [ax1, ax2]:
        ax.plot(x_vals, y_vals, 'r-', linewidth=2.5, label='Truncation Boundary', zorder=4)
    
    # Show the valid region with a clearer representation
    # Create a polygon for the valid region
    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]
    
    corners = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min]
    ])
    
    # Determine which side is valid based on a representative point
    # Assume a^T x <= b defines the valid region
    valid_region = []
    for corner in corners:
        if np.dot(a, corner) <= b:  # Check if point is in valid region
            valid_region.append(corner)
    
    # Add intersection points of truncation line with plot boundaries
    if len(x_vals) >= 2:
        for i in range(len(x_vals)):
            valid_region.append([x_vals[i], y_vals[i]])
    
    # If we have enough points, create a polygon
    if len(valid_region) >= 3:
        from scipy.spatial import ConvexHull
        valid_region = np.array(valid_region)
        hull = ConvexHull(valid_region)
        valid_region = valid_region[hull.vertices]
        
        # Plot the polygon
        for ax, alpha in zip([ax1, ax2], [0.15, 0.25]):  # More visible in truncated plot
            polygon = Polygon(valid_region, facecolor='green', edgecolor='none', 
                             alpha=alpha, label='Valid Region', zorder=1)
            ax.add_patch(polygon)
    
    # Add colorbar and labels with improved styling
    cbar1 = fig.colorbar(cs1, ax=ax1)
    cbar2 = fig.colorbar(cs2, ax=ax2)
    cbar1.set_label('Probability Density', fontsize=12)
    cbar2.set_label('Probability Density', fontsize=12)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        ax.legend(loc='upper right', framealpha=0.9)
        # Set equal aspect to avoid distortion
        ax.set_aspect('equal', adjustable='box')
        
        # Add tick labels
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig

# 测试案例
mu_0 = np.array([1.0, 2.0])  # 均值向量
Sigma_0 = np.array([[1.0, 0.5], 
                    [0.5, 2.0]])  # 协方差矩阵
a = np.array([1.0, -1.0])  # 法向量
b = 0.5  # 位置参数

mu_trunc, Sigma_trunc = BLTruncation(mu_0, Sigma_0, a, b)

print("原始均值向量:")
print(mu_0)
print("\n原始协方差矩阵:")
print(Sigma_0)
print("\n截断后的均值向量:")
print(mu_trunc)
print("\n截断后的协方差矩阵:")
print(Sigma_trunc)

# # 可视化
# fig = visualize_truncation(mu_0, Sigma_0, mu_trunc, Sigma_trunc, a, b)
# plt.show()

