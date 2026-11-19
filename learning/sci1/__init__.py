"""
SCI-1: coverage under blind-corner occlusion with budgeted random verification.

Everything in this package is ROS-independent (NumPy only, Matplotlib for
plots). The 2D simulator, the shared decision-log format, the logging and
verification policies, the conformal calibrators and the planners are the
same code that will later sit behind thin ROS wrappers for Gazebo and the
real robot; only the data source changes.
"""
