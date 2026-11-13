"""Make every figure of the paper from data/ (written by ../build.py).

    python make_all.py
"""
import plot_omega
import plot_profile2d
import plot_profile_triangle
import plot_scan
import plot_triangle

for module in (plot_triangle, plot_omega, plot_profile2d, plot_profile_triangle, plot_scan):
    module.main()
