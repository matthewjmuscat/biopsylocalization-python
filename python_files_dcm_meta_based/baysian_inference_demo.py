import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation

def one_plot_animation(gif_name, data, alpha_prior=1, beta_prior=1, fps=10):
    # Fake data: 1 represents success, 0 represents failure
    true_p = 0.7  # True probability of success
    
    n_trials = len(data)

    fig, ax = plt.subplots()

    # Setting up the plot limits and labels
    y_top = 1.1
    x = np.linspace(0, 1, 100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_top)  # Adjusted for normalized plot
    line, = ax.plot(x, beta.pdf(x, alpha_prior, beta_prior) / beta.pdf(x, alpha_prior, beta_prior).max(), 'r-')
    ax.set_title('Beta-Binomial Bayesian model of tissue class probability')
    ax.set_xlabel('Probability of success')
    ax.set_ylabel('Density')

    # Adding text annotation for true prior, number of trials, equation updates, and play status
    true_prior_text = ax.text(0.02, y_top-0.05, f'True p: {true_p}', fontsize=9, ha='left')
    trial_number_text = ax.text(0.02, y_top-0.1, 'n: 0', fontsize=9, ha='left')
    prior_text = ax.text(0.02, y_top-0.15, f'Prior: Beta({alpha_prior}, {beta_prior})', fontsize=9, ha='left')
    posterior_text = ax.text(0.02, y_top-0.2, f'Posterior: Beta({alpha_prior}, {beta_prior})', fontsize=9, ha='left')
    expectation_text = ax.text(0.02, y_top-0.25, '', fontsize=9, ha='left')
    std_dev_text = ax.text(0.02, y_top-0.3, '', fontsize=9, ha='left')
    status_text = ax.text(0.98, y_top-0.05, 'Paused', fontsize=9, ha='right')

    def update(frame):
        nonlocal alpha_prior, beta_prior
        status = 'Playing'
        if frame < 20 or frame >= n_trials + 20:
            status = 'Paused'  # Pause during the first and last 20 frames
        elif 20 <= frame < n_trials + 20:
            # Update the parameters of the beta distribution
            alpha_prior += data[frame - 20]
            beta_prior += 1 - data[frame - 20]
            trial_number_text.set_text(f'n: {frame - 19}')  # Update n only after delay

        # Update the plot with normalized maximum value
        y = beta.pdf(x, alpha_prior, beta_prior)
        line.set_ydata(y / y.max())
        posterior_text.set_text(f'Posterior: Beta({alpha_prior}, {beta_prior})')
        status_text.set_text(status)
        # Calculate expectation and standard deviation
        expectation = alpha_prior / (alpha_prior + beta_prior)
        std_dev = np.sqrt(alpha_prior * beta_prior / ((alpha_prior + beta_prior) ** 2 * (alpha_prior + beta_prior + 1)))
        expectation_text.set_text(f'Expectation: {expectation:.3f}')
        std_dev_text.set_text(f'Std. Deviation: {std_dev:.3f}')

        return line, title, trial_number_text, true_prior_text, prior_text, posterior_text, status_text, expectation_text, std_dev_text

    # Creating the animation
    ani = FuncAnimation(fig, update, frames=n_trials + 40, blit=True)

    # Save the animation as a gif
    ani.save(f'{gif_name}.gif', writer='imagemagick', fps=fps)

    plt.show()

n_trials = 100
data = np.random.binomial(1, true_p, size=n_trials)

arg_tuple = (('gif_1', data, 1, 1), ('gif_2', data, 50, 1), ('gif_3', data, 1, 50))

for args in arg_tuple:
    one_plot_animation(*args)
