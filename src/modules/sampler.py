import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.num_steps, self.beta_start, self.beta_end = num_steps, beta_start, beta_end
        self.register_buffer("betas", self.get_linear_schedule(), persistent=False)
        self.register_buffer("alphas", 1 - self.betas, persistent=False)
        self.register_buffer(
            "barred_alphas", torch.cumprod(self.alphas, dim=-1), persistent=False
        )

    def get_linear_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps)

    def add_noise(self, images, timesteps):
        """
        add noise by doing forward diffusion
        compute the distribution of noise at `timesteps`
        i.e., compute mu and sigma after adding `timesteps` of noises
        see equation (4) of https://arxiv.org/abs/2006.11239

        params:
            images - (batch_size, channels, height, width)
            timesteps - (batch_size, )
        """

        barred_alpha_t = self.barred_alphas[timesteps]
        mean_coeff = torch.sqrt(barred_alpha_t)
        exp_dim = tuple([-1] + [1 for _ in range(images.dim() - 1)])
        mu = mean_coeff.view(exp_dim) * images

        sigma = (1.0 - barred_alpha_t).sqrt()

        noise = torch.randn_like(images)
        noisy_images = mu + sigma.view(exp_dim) * noise
        return noisy_images, noise

    def remove_noise(self, noisy_images, timesteps, predicted_noise):
        """
        remove noise by doing reverse diffusion
        see Algorithm 2 of https://arxiv.org/abs/2006.11239
        By DDPM paper, sigma_sq_t = beta_t and sigma_sq_t = beta_t * (1 - barred_alpha_{t-1}) / (1 - barred_alpha_t)
        give similar result. I will use the latter.
        params:
            noisy_images - (batch_size, channels, height, width)
            timesteps - (batch_size, )
            predicted_noise - (batch_size, channels, height, width)
        """

        barred_alpha_t = self.barred_alphas[timesteps]

        barred_alpha_prev_t = self.barred_alphas[timesteps - 1]
        equal_zero_mask = timesteps == 0
        barred_alpha_prev_t[equal_zero_mask] = 1

        beta_t = self.betas[timesteps]
        sigma_t = torch.sqrt(beta_t * (1 - barred_alpha_prev_t) / (1 - barred_alpha_t))

        noise_coeff = beta_t / (1 - barred_alpha_t).pow(0.5)

        inv_sqrt_alpha_t = self.alphas[timesteps].pow(-0.5)

        exp_dim = tuple([-1] + [1 for _ in range(noisy_images.dim() - 1)])
        mu = inv_sqrt_alpha_t.view(exp_dim) * (
            noisy_images - noise_coeff.view(exp_dim) * predicted_noise
        )

        z = torch.randn_like(predicted_noise)
        denoised = mu + sigma_t.view(exp_dim) * z

        return denoised


if __name__ == "__main__":
    sampler = Sampler()
    rand = torch.randn(4, 3, 32, 32)
    timesteps = torch.randint(0, sampler.num_steps, (4,))
    pred_noise = torch.randn_like(rand)
    sampler.add_noise(rand, timesteps)
    sampler.remove_noise(rand, timesteps, pred_noise)
