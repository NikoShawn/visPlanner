#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

namespace BLTrunc {

// Constants for numerical precision
constexpr double SQRT_2PI = 2.50662827463;
constexpr double SQRT_2 = 1.41421356237;

// Helper function for standard normal PDF
double normalPDF(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

// Helper function for standard normal CDF using error function
double normalCDF(double x) {
    return 0.5 * (1 + std::erf(x / SQRT_2));
}

/**
 * Barber and Lopez-Perez Truncation algorithm
 * @param mu_0 Original mean vector
 * @param Sigma_0 Original covariance matrix
 * @param a Normal vector of truncation plane (will be converted to column vector)
 * @param b Position parameter of truncation plane
 * @return std::pair containing the truncated mean vector and covariance matrix
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> BLTruncation(
    const Eigen::VectorXd& mu_0,
    const Eigen::MatrixXd& Sigma_0,
    Eigen::VectorXd a,
    double b) {
    
    // Reverse direction of the normal vector
    a = -a;
    b = -b;
    
    // Calculate key parameters
    double q0 = a.transpose() * Sigma_0 * a;  // Scalar result
    double alpha = (b - a.transpose() * mu_0) / std::sqrt(q0);
    
    // Calculate truncation parameter
    double lambda_val = normalPDF(alpha) / normalCDF(alpha);
    
    // Handle numerical issues
    if (std::isnan(lambda_val)) {
        lambda_val = -alpha;
    }
    
    // Update the distribution parameters
    double mu1 = a.transpose() * mu_0 - lambda_val * std::sqrt(q0);
    double var1 = q0 * (1 - lambda_val * lambda_val - alpha * lambda_val);
    
    // Calculate mean shift
    Eigen::VectorXd dmu = (Sigma_0 * a) * ((a.transpose() * mu_0 - mu1) / q0);
    
    // Update mean and covariance
    Eigen::VectorXd mu = mu_0 - dmu;
    
    // Calculate matrix term
    Eigen::MatrixXd term = (Sigma_0 * a) * (a.transpose() * Sigma_0);
    Eigen::MatrixXd Sigma = Sigma_0 - term * ((q0 - var1) / (q0 * q0));
    
    return std::make_pair(mu, Sigma);
}

/**
 * Struct to hold truncated distribution information
 */
struct TruncatedDistribution {
    Eigen::VectorXd mu;               // Mean vector
    Eigen::MatrixXd Sigma;            // Covariance matrix
    Eigen::VectorXd boundaryNormal;   // Normal vector (may be null)
    double boundaryPosition;          // Position parameter (may be invalid)
    std::string label;                // Label for the distribution
    
    TruncatedDistribution(
        const Eigen::VectorXd& mu, 
        const Eigen::MatrixXd& Sigma,
        const Eigen::VectorXd& normal = Eigen::VectorXd(),
        double position = 0.0,
        const std::string& label = ""
    ) : mu(mu), Sigma(Sigma), boundaryNormal(normal), 
        boundaryPosition(position), label(label) {}
};

/**
 * Apply individual truncations
 * @param mu_0 Original mean vector
 * @param Sigma_0 Original covariance matrix
 * @param A Matrix of normal vectors (each row is a normal vector)
 * @param b Vector of position parameters
 * @return Vector of TruncatedDistribution objects
 */
std::vector<TruncatedDistribution> IndividualTruncations(
    const Eigen::VectorXd& mu_0,
    const Eigen::MatrixXd& Sigma_0,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b) {
    
    std::vector<TruncatedDistribution> truncated_distributions;
    
    // Add the original distribution
    truncated_distributions.emplace_back(mu_0, Sigma_0, Eigen::VectorXd(), 0.0, "Original");
    
    // Apply truncation for each boundary
    for (int i = 0; i < b.size(); i++) {
        Eigen::VectorXd a_i = A.row(i);
        double b_i = b(i);
        
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> result = BLTruncation(mu_0, Sigma_0, a_i, b_i);
        Eigen::VectorXd mu_i = result.first;
        Eigen::MatrixXd Sigma_i = result.second;
        
        truncated_distributions.emplace_back(
            mu_i, Sigma_i, a_i, b_i, "Boundary " + std::to_string(i+1)
        );
    }
    
    return truncated_distributions;
}

} // namespace BLTrunc