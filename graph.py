import matplotlib.pyplot as plt

# Re-plot with stylized design and annotations
plt.figure(figsize=(10, 7))
plt.quiver(X, Y, dX_norm, dY_norm, magnitude, cmap='plasma', scale=30)
plt.xlabel('Factor Endowment ($x_i$)', fontsize=12)
plt.ylabel('Asymmetric Growth ($y_i$)', fontsize=12)
plt.title('Phase Portrait of Pareto-Nash Strategic Dynamics', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.colorbar(label='Vector Magnitude')

# Annotate the stable core region
plt.scatter([0.6], [0.6], color='red', s=100, label='Core Allocation (Stable Equilibrium)')
plt.text(0.65, 0.65, 'Stable Core', fontsize=10, color='red')

# Annotate an unstable trajectory (saddle path)
plt.arrow(1.0, 1.0, 0.3, 0.3, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
plt.text(1.15, 1.35, 'Saddle Path (Defection)', fontsize=10, color='blue')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
