noise = np.random.normal(0, 0.1)  # Gaussian noise with mean 0 and std dev 0.1
        external_force = 1.0 + noise
        
        y = hummingbird.update(pwm, external_force)  # Pr