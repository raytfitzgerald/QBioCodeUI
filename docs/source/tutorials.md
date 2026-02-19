# Tutorials

Welcome to the QBioCode tutorials! These Jupyter notebooks provide hands-on examples demonstrating how to use various features and applications of QBioCode for quantum healthcare and life sciences applications.

## Getting Started

Before running these tutorials, make sure you have:
- Installed QBioCode following the [Installation Guide](installation.md)
- Set up your Python environment with all required dependencies
- Access to quantum computing resources (if running quantum algorithms)

---

## Tutorial Gallery

### 1. Artificial Data Generation

Learn how to generate synthetic datasets for testing and benchmarking quantum machine learning algorithms.

<a href="tutorials/Artificial_data_generation/example_data_generation.html">📓 <strong>View Tutorial Notebook</strong></a>

---

### 2. QProfiler - Automated ML Model Benchmarking

Learn how to use QProfiler to systematically benchmark and compare quantum and classical machine learning models on artificial datasets. This tutorial demonstrates:

<a href="tutorials/QProfiler/example_qprofiler.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate artificial datasets with specific characteristics
- Configure and run QProfiler experiments via YAML configuration
- Evaluate multiple ML models (quantum and classical) automatically
- Analyze performance metrics (accuracy, F1-score, AUC)
- Visualize model comparisons and correlations
- Interpret results for model selection

---

### 3. QSage - Quantum-Inspired Feature Importance

Explore QSage, an intelligent meta-learning system that predicts which machine learning models will perform best on your dataset *before* you run them. By learning from data complexity patterns across multiple datasets, QSage provides data-driven model recommendations. This tutorial shows how to:

<a href="tutorials/QSage/qsage.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Load pre-trained QSage models
- Analyze dataset characteristics (intrinsic dimension, Fisher discriminant ratio, etc.) from compiled ML benchmark results
- Apply QSAGE to predict the model

---

### 4. Quantum Projection Learning (QPL)

Learn about Quantum Projection Learning techniques and their applications in machine learning workflows.

<a href="tutorials/Quantum_Projection_Learning/QPL_example.html">📓 <strong>View Tutorial Notebook</strong></a>

**Topics Covered:**
- Quantum projection methods
- Integration with classical ML pipelines
- Performance optimization

---

### 5. Projected Quantum Kernel (PQK)

Explore the Projected Quantum Kernel approach for quantum machine learning.

<a href="tutorials/PQK%20-%20OV.html">📓 <strong>View Tutorial Notebook</strong></a>

**Key Concepts:**
- Quantum kernel methods
- Kernel-based classification
- Quantum advantage in kernel computation

---

## Additional Resources

- [API Documentation](api_overview.rst) - Detailed API reference
- [QProfiler App](apps/profiler.rst) - Standalone profiling application
- [QSage App](apps/sage.rst) - Feature selection application
- [GitHub Repository](https://github.com/IBM/QBioCode) - Source code and examples

## Support

If you encounter any issues or have questions about the tutorials:
- Check the [GitHub Issues](https://github.com/IBM/QBioCode/issues)
- Review the [Contributing Guide](https://github.com/IBM/QBioCode/blob/main/CONTRIBUTING.md)
- Consult the API documentation for detailed function references
