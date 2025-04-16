import numpy as np
import torch
import torch.nn as nn
from torch.distributions import VonMises
from typing import *

from unifold.modules.frame import Frame, Rotation

from ml_collections import ConfigDict

from . import so3
from .angle import IGSO3


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().float().numpy()

class EuclideanDiffuser(nn.Module):
    def __init__(
        self,
        scale_factor: float = 1.,
        beta_clip: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scale_factor = scale_factor or 1.
        self.beta_clip = beta_clip
        self.eps = eps
    
    def beta(self, t, s):
        assert torch.all(t > s)
        return 1. -self.gamma(t) / self.gamma(s)

    def gamma(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        '''
        mapping from t to gamma_t. Make sure gamma(0) = 1 and gamma(1) ~= 0.
        '''
        return 1. - t

    def prior(
        self,
        shape,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        # print("pos prior", np.random.get_state())
        z = np.random.randn(*shape)
        z = torch.tensor(z, dtype=dtype, device=device)
        return z * self.scale_factor

    def addnoise(
        self,
        x_s: torch.Tensor,          # [*, L, D]
        diff_mask: torch.Tensor,    # [*, L]
        t: torch.Tensor,            # [*]
        s: torch.Tensor = None,     # [*]
    ):
        '''
        forward transition from timestamp `s` to `t` (s < t).
        '''
        if s is None:   # by default s=0 and gamma(s)=1.
            s = torch.zeros_like(t)
        else:
            assert torch.all(s < t)

        # print("pos addn", np.random.get_state())
        z = np.random.randn(*x_s.shape)
        z = torch.tensor(z, dtype=x_s.dtype, device=x_s.device)

        g = self.gamma(t) / self.gamma(s)

        x_t = g.sqrt() * x_s + (1. - g).sqrt() * z * self.scale_factor
        if diff_mask is not None:
            x_t = torch.where(
                diff_mask[..., None] > 0,
                x_t,
                x_s
            )

        return x_t, z
    
    def denoise(
        self,
        x_t: torch.Tensor,          # [*, L, D]
        xh_0: torch.Tensor,          # [*, L, D]
        diff_mask: torch.Tensor,    # [*, L]
        t: torch.Tensor,            # [*]
        s: torch.Tensor,     # [*]
        res1_indices,
        res2_indices,
        cutoff_list,
        eta = 0.01,
        g_decay = 5,
        decay_radius = 0, 
        decay_lambda = 0.1
    ):
        '''
        backward transition from timestamp `t` to `s` (s < t).
        '''
        assert torch.all(s < t)

        z = np.random.randn(*x_t.shape)
        z = torch.tensor(z, dtype=x_t.dtype, device=x_t.device)

        gt, gs = self.gamma(t), self.gamma(s)

        beta = (1. - gt / gs)[..., None].clamp_max(self.beta_clip)

        score = self.score(x_t, xh_0, t)

        # x_s = (2. - (1. - beta).sqrt()) * x_t + beta * score + beta.sqrt() * z * self.scale_factor
        x_s = (1/(1 - beta).sqrt())*( x_t + beta * score * self.scale_factor**2) + ((1 - gs)/(1 - gt)*beta).sqrt() * z * self.scale_factor
        for idx1, idx2 in zip(res1_indices,res2_indices):
            x1, x2 = x_s[idx1], x_s[idx2]
            # print(f"{idx1} pos is original x1 {x1}")
            # print(f"{idx2} pos is original x2 {x2}")
            # Compute the guidance vector
            dist_current = torch.norm(x2 - x1, dim=-1, keepdim=True)
            # print("original dist",dist_current)
        # Compute the guidance vector according to x_s rather than x_t, bc x_s is the one we want to update
        guidance =  self.compute_guidance(x_s, res1_indices, res2_indices, cutoff_list,decay_radius, decay_lambda)
        # print("beta score",beta * score * self.scale_factor**2)
        # print("eta guidance", eta * guidance[guidance >0])
        # print("g_decay", g_decay)
        # print("eta",eta)
        # x_s = x_s + torch.cos(torch.pi/2 * (t**g_decay - 1)) * eta * guidance 
        x_s = x_s + eta * guidance 
        for idx1, idx2 in zip(res1_indices,res2_indices):
            x1, x2 = x_s[idx1], x_s[idx2]
            # print(f"{idx1} pos is guided x1 {x1}")
            # print(f"{idx2} pos is guided x2 {x2}")
            # Compute the guidance vector
            dist_current = torch.norm(x2 - x1, dim=-1, keepdim=True)
            # print("guided dist",dist_current)
        # x_s = torch.where(
        #     s[..., None, None] > 1e-12, x_s, xh_0
        # )

        if diff_mask is not None:
            x_s = torch.where(
                diff_mask[..., None] > 0,
                x_s,
                x_t
            )

        return x_s, beta

    def score(
        self,
        x_t: torch.Tensor,          # [*, L, D]
        x_0: torch.Tensor,          # [*, L, D]
        t: torch.Tensor,            # [*]
    ):
        '''
        stein score function.
        '''
        g = self.gamma(t)[..., None, None]
        score = (g.sqrt() * x_0 - x_t) / (1. - g) / self.scale_factor**2
        return score
    
    def compute_guidance(self, x_t, res1_indices, res2_indices, cutoff_list, decay_radius=0, decay_lambda=0.1):
        """
        Compute a smooth, propagated guidance term for diffusion, based on atom pair constraints
        and their neighboring residues with exponential decay, using relative directions for neighboring residue pairs.
        Additionally, for each res1_index, search for the largest distance pair with res2_index within the offset range
        and apply guidance calculation.

        Args:
            x_t (torch.Tensor): Current positions of shape [L, 3], where L is the number of residues/atoms.
            res1_indices (list[int]): List of first residue indices in constraints.
            res2_indices (list[int]): List of second residue indices in constraints.
            cutoff_list (list[float]): List of desired upper-bound distance between residue pairs.
            decay_radius (int): Number of neighboring residues on each side to apply decayed guidance.
            decay_lambda (float): Decay rate for exponential weight: exp(-lambda * d), where d is sequence distance.

        Returns:
            torch.Tensor: A tensor of shape [L, 3] representing the smooth guidance vectors.
        """
        L = x_t.shape[0]
        G = torch.zeros_like(x_t)

        for idx1, idx2, cutoff in zip(res1_indices, res2_indices, cutoff_list):
            # Check the distance between the two residues
            x1, x2 = x_t[idx1], x_t[idx2]
            dist = torch.norm(x2 - x1).clamp(min=1e-6)

            # If distance is smaller than cutoff, no guidance needed
            if dist < cutoff:
                continue

            delta = dist - cutoff

            # Loop through the neighboring indices for res2 within the offset range
            for offset in range(-decay_radius, decay_radius + 1):
                # Initialize variables to track the maximum distance pair
                max_dist = -1
                max_idx1 = None

                # Find the neighboring index for res2 within the range of res1
                n_idx2 = idx2 + offset

                # Ensure n_idx2 is within bounds and not the same as idx1
                if 0 <= n_idx2 < L and n_idx2 != idx1:
                    # Compute the distance to the neighboring residue
                    dist_to_res2 = torch.norm(x_t[idx1] - x_t[n_idx2])
                    
                    # Find the largest distance to idx2
                    if dist_to_res2 > max_dist:
                        max_dist = dist_to_res2
                        max_idx1 = idx1
            
                    # If a valid neighboring index was found, compute the guidance
                    if max_idx1 is not None and max_dist > cutoff:
                        # Compute the exponential decay weight for the guidance
                        w = torch.exp(-torch.tensor(decay_lambda) * abs(offset))

                        # Compute the relative direction for the larger distance pair
                        relative_direction = (x_t[n_idx2] - x_t[max_idx1]) / torch.norm (x_t[n_idx2] - x_t[max_idx1]).clamp(min=1e-6)

                        # Apply guidance based on the largest distance pair
                        G[n_idx2] = -delta * w * relative_direction

        return G


class PositionDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        scale_factor: float,
        kernel: str,
        params: Tuple[float],
        beta_clip: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(scale_factor, beta_clip, eps)
        self.kernel = kernel
        self.params = params

    def gamma(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        '''
        mapping from t to gamma_t. Make sure gamma(0) = 1 and gamma(1) = 0.
        '''
        if self.kernel == "exp":
            k, a = self.params
            return torch.exp(-k * (t ** a))
        elif self.kernel == "ddpm":
            s, a = self.params
            return improved_ddpm_kernel(t ** a, s, T=1)
        elif self.kernel == "linear":
            a, b, x, y = self.params
            return torch.where(
                t < x, 1.-(t/x)**a*(1.-y), y*((1.-t)/(1.-x)) ** b
            )

def improved_ddpm_kernel(t, s, T=1):
    s = torch.tensor(s)
    fn = lambda x: torch.cos(x * torch.pi / 2) ** 2
    normed_t = (t + s) / (T + s)
    normed_zero = s / (T + s)
    return fn(normed_t) / fn(normed_zero)


def _shape_to_size(shape):
    n = 1
    for i in shape:
        n *= i
    return n

def uniform_random_rotations(
    shape: Iterable,
    dtype: torch.dtype = torch.float,
    device: torch.device = "cpu",
) -> torch.Tensor:
    np_random_rots = so3.random(_shape_to_size(shape))
    random_rots = torch.tensor(
        np_random_rots, dtype=dtype, device=device
    ).view(*shape, 3, 3)
    return random_rots


class RotationDiffuser(nn.Module):
    def __init__(
        self,
        max_sigma_sq: float = 9.,
        coef_a: float = 1.,
        rw_approx_thres: float = 0.6,
        left: bool = True,
        num_omega_bins: int = 1000,
        l_cutoff: int = 100,
        igso3_gaussian_thres: float = 0.2,
        igso3_grad_gaussian_thres: float = 0.6,
        eps: float = 1e-12,
    ):
        # approx_thres:
        #   for sigma > approx_thres we use IGSO3 angle distribution.
        #   for sigma < approx_thres we use Brownian motion in the tangent space.
        super().__init__()
        self.max_sigma_sq = max_sigma_sq
        self.coef_a = coef_a
        self.rw_approx_thres = rw_approx_thres
        self.left = left
        self.igso3 = IGSO3(
            num_omega_bins=num_omega_bins,
            l_cutoff=l_cutoff,
            gaussian_thres=igso3_gaussian_thres,
            grad_gaussian_thres=igso3_grad_gaussian_thres,
            eps=eps,
        )

    @staticmethod
    def prior(
        shape,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        # print("prior rot", np.random.get_state())
        return uniform_random_rotations(shape, dtype, device)

    def sigma_sq(self, t: torch.Tensor):
        '''
        compute sigma^2_t.
        must make sigma(0) = 0.
        making sigma(T=1) >= 3 guarantees that R_T=1 is from a uniform so3 prior.
        '''
        return self.max_sigma_sq * (t ** self.coef_a)

    def addnoise(
        self,
        r_s: torch.Tensor,          # * L 3 3
        diff_mask: torch.Tensor,    # * L
        t: torch.Tensor,            # *
        s: torch.Tensor = None,     # *
    ):
        if s is None:
            s = torch.zeros_like(t)
        else:
            assert torch.all(s <= t), (s, t)

        betas = self.sigma_sq(t) - self.sigma_sq(s) # *
        delta_r = self.random_rotations(
            sigmas=betas.sqrt(),
            size=r_s.shape[-3],
            igso3_obj=self.igso3,
            rw_approx_thres=self.rw_approx_thres,
        )   # * L 3 3

        delta_r = delta_r.to(device=r_s.device, dtype=r_s.dtype)
        r_t = delta_r @ r_s if self.left else r_s @ delta_r

        if diff_mask is not None:
            r_t = torch.where(diff_mask[..., None, None] > 0, r_t, r_s)

        return r_t, delta_r

    def score(
        self,
        r_t: torch.Tensor,      # [*, L, 3, 3]
        rh_0: torch.Tensor,     # [*, L, 3, 3]
        sigma_t: torch.Tensor,        # [*]
    ) -> torch.Tensor:          # [*, L, 3]

        if self.left:
            r_z = r_t @ rh_0.transpose(-1, -2)  # * L 3 3
        else:
            r_z = rh_0.transpose(-1, -2) @ r_t

        v_z = self.Log(r_z) # * L 3
        rw_approx = -v_z / sigma_t[..., None, None] ** 2
        if torch.all(sigma_t < self.rw_approx_thres):
            return rw_approx

        w = v_z.norm(dim=-1)  # * L
        dlogf_dw = self.igso3.grad_log_pdf(sigma_t, w)  # * L

        score = torch.where(
            sigma_t[..., None, None] > self.rw_approx_thres,
            v_z * dlogf_dw[..., None] / w[..., None], rw_approx
        )

        return score

    def denoise(
        self,
        r_t: torch.Tensor,      # * L 3 3
        rh_0: torch.Tensor,     # * L 3 3
        diff_mask: torch.Tensor,# * L
        t: torch.Tensor,        # *
        s: torch.Tensor,        # *
    ):
        assert torch.all(s <= t), (s, t)
        sigma_sqs_t = self.sigma_sq(t)

        betas = sigma_sqs_t - self.sigma_sq(s)
        z_so3 = self.random_so3(
            sigmas=betas.sqrt(),
            size=r_t.shape[-3],
            igso3_obj=self.igso3,
            rw_approx_thres=self.rw_approx_thres,
        )   # * L 3
        z_so3 = torch.where(
            s[..., None, None] > 0, z_so3, z_so3.new_zeros(*z_so3.shape)
        )

        score = self.score(r_t, rh_0, sigma_sqs_t.sqrt())   # * L 3
        update_vec = score * betas[..., None, None]
        noisy_update_vec = update_vec + z_so3
        update_mat = self.Exp(noisy_update_vec)

        if self.left:
            r_s = update_mat @ r_t
        else:
            r_s = r_t @ update_mat

        if diff_mask is not None:
            r_s = torch.where(
                diff_mask[..., None, None] > 0.,
                r_s, r_t
            )

        return r_s, update_vec
    
    def denoise_sigma(
        self,
        r_t: torch.Tensor,      # * L 3 3
        rh_0: torch.Tensor,     # * L 3 3
        diff_mask: torch.Tensor,# * L
        sigma_sqs_t: torch.Tensor,        # *
        sigma_sqs_s: torch.Tensor,        # *
    ):
        betas = sigma_sqs_t - sigma_sqs_s
        delta_so3 = self.random_so3(
            sigmas=betas.sqrt(),
            size=r_t.shape[-3],
            igso3_obj=self.igso3,
            rw_approx_thres=self.rw_approx_thres,
        )   # * L 3
        # delta_so3 = torch.where(
        #     s[..., None, None] > 0, delta_so3, delta_so3.new_zeros(*delta_so3.shape)
        # )

        score = self.score(r_t, rh_0, sigma_sqs_t.sqrt())   # * L 3
        update_vec = score * betas[..., None, None]
        noisy_update_vec = update_vec + delta_so3
        update_mat = self.Exp(noisy_update_vec)

        if self.left:
            r_s = update_mat @ r_t
        else:
            r_s = r_t @ update_mat

        if diff_mask is not None:
            r_s = torch.where(
                diff_mask[..., None, None] > 0.,
                r_s, r_t
            )

        return r_s, update_vec

    @staticmethod
    def Log(r: torch.Tensor):
        batch_shape = r.shape[:-2]
        np_r = to_numpy(r.reshape(-1, 3, 3))
        np_v = so3.Log(np_r)
        v = r.new_tensor(np_v).view(*batch_shape, 3)
        return v

    @staticmethod
    def Exp(v: torch.Tensor):
        batch_shape = v.shape[:-1]
        np_v = to_numpy(v.reshape(-1, 3))
        np_r = so3.Exp(np_v)
        r = v.new_tensor(np_r).view(*batch_shape, 3, 3)
        return r

    @staticmethod
    def random_so3(
        sigmas: torch.Tensor,   # *
        size: int,      # N=size
        igso3_obj: IGSO3 = None,
        rw_approx_thres: float = 0.6,
    ) -> torch.Tensor:
        # print("rand so3", np.random.get_state())
        if igso3_obj is None:
            print("warning: igso3 obj not provided.")
            igso3_obj = IGSO3()

        z = sigmas.new_tensor(np.random.randn(*sigmas.shape, size, 3))   # * N 3
        rotvec_approx = z * sigmas[..., None, None]
        if torch.all(sigmas <= rw_approx_thres):
            return rotvec_approx

        angles = igso3_obj.sample(sigmas, size)     # * N
        rotvec = torch.where(
            sigmas[..., None, None] > rw_approx_thres,
            z * angles[..., None] / z.norm(dim=-1, keepdim=True),
            rotvec_approx,
        )

        return rotvec

    @staticmethod
    def random_rotations(
        sigmas: torch.Tensor,   # *
        size: int,      # N=size
        igso3_obj: IGSO3 = None,
        rw_approx_thres: float = 0.6,
    ) -> torch.Tensor:
        rotvec = RotationDiffuser.random_so3(sigmas, size, igso3_obj, rw_approx_thres)
        rotmat = RotationDiffuser.Exp(rotvec)
        return rotmat


TWOPI = 2*torch.pi
class ChiAngleDiffuser(nn.Module):
    def __init__(
        self,
        kernel,
        params,
        n_approx: int = 10,
        enabled: bool = True
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.params = params

        self.periods = torch.arange(-n_approx, n_approx+1, 1) * TWOPI

    @staticmethod
    def prior(
        shape, dtype, device
    ):
        return torch.tensor(
            np.random.rand(*shape) * TWOPI,
            dtype=dtype, device=device
        )

    def sigma_sq(self, t: torch.Tensor):
        '''
        compute sigma^2_t.
        must make sigma(0) = 0.
        making sigma(T=1) >= 3 guarantees that R_T=1 is from a uniform so3 prior.
        '''
        if self.kernel == "exp":
            max_sigma_sq, coef_a = self.params
            sigma_sq =  max_sigma_sq * (t ** coef_a)
        else:
            raise ValueError(f"unknown kernel name {self.kernel}")
        return sigma_sq

    def addnoise(
        self,
        a_s: torch.Tensor,          # * L 4
        diff_mask: torch.Tensor,    # * L
        t: torch.Tensor,            # *
        s: torch.Tensor = None,     # *
    ):
        # print("tor addn", np.random.get_state())
        sigma_sq_t = self.sigma_sq(t)
        sigma_sq_s = 0 if s is None else self.sigma_sq(s)
        betas = sigma_sq_t - sigma_sq_s
        z = torch.tensor(np.random.randn(*a_s.shape)).to(device=a_s.device, dtype=a_s.dtype)
        # vm = VonMises(loc=torch.zeros_like(betas), concentration=1./betas)
        # z = vm.sample(a_s.shape)
        a_t = (a_s + betas.sqrt()[..., None, None] * z) % (TWOPI)
        a_t = torch.where(diff_mask[..., None] > 0., a_t, a_s)
        return a_t

    def score(
        self,
        a_t: torch.Tensor,          # [*, L, 4]
        a_0: torch.Tensor,          # [*, L, 4]
        sigma_sq: torch.Tensor,            # [*]
    ):
        '''
        stein score function.
        '''
        from unicore.utils import batched_gather
        diff = (a_t - a_0)[..., None] + self.periods.to(device=a_t.device, dtype=a_t.dtype)
        exp = torch.exp(-.5 * diff**2 / sigma_sq[..., None, None])
        lin = -diff / sigma_sq[..., None, None]
        ret = (lin * exp).sum(-1) / (exp.sum(-1) + 1e-8)
        return ret

    def denoise(
        self,
        a_t: torch.Tensor,      # * L 4
        ah_0: torch.Tensor,     # * L 4
        diff_mask: torch.Tensor,# * L
        t: torch.Tensor,        # *
        s: torch.Tensor,        # *
    ):
        assert torch.all(t > s)
        sigma_sq = self.sigma_sq(t)
        score = self.score(a_t, ah_0, sigma_sq)
        betas = (sigma_sq - self.sigma_sq(s))[..., None, None]
        z = torch.tensor(np.random.randn(*a_t.shape)).to(device=a_t.device, dtype=a_t.dtype)
        # vm = VonMises(loc=torch.zeros_like(betas), concentration=1./betas)
        # z = vm.sample(a_t.shape)
        a_s = (a_t + score * betas + z * betas.sqrt()) % TWOPI
        a_s = torch.where(
            s[..., None, None] < 1e-6,
            ah_0, a_s
        )
        a_s = torch.where(
            diff_mask[..., None] > 0.5,
            a_s, a_t
        )
        return a_s


def sin_cos_to_angles(sin_cos: torch.Tensor, use_atan2=True):
    s, c = torch.chunk(sin_cos, 2, dim=-1)
    if use_atan2:
        ret = torch.atan2(s, c) % TWOPI
    else:
        ret = (torch.asin(s) * torch.sign(c)) % TWOPI
    return ret.squeeze(-1)

def angles_to_sin_cos(angles):
    return torch.stack([angles.sin(), angles.cos()], dim=-1)


def frames_to_r_p(f: torch.Tensor):
    r, p = torch.split(f[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
    p = p.squeeze(-1)
    return r, p

def r_p_to_frames(r: torch.Tensor, p: torch.Tensor):
    return Frame(Rotation(r), p).to_tensor_4x4()

class Diffuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_trans = PositionDiffuser(**config.position)
        self.rot_trans = RotationDiffuser(**config.rotation)
        self.chi_trans = None
        if config.chi.enabled:
            self.chi_trans = ChiAngleDiffuser(**config.chi)

    def prior(
        self,
        batch_shape: Tuple[int],
        seq_len: int,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ):
        r_T = self.rot_trans.prior((*batch_shape, seq_len), dtype, device)
        p_T = self.pos_trans.prior((*batch_shape, seq_len, 3), dtype, device)
        if self.chi_trans:
            a_T = self.chi_trans.prior((*batch_shape, seq_len, 4), dtype, device)
            return r_p_to_frames(r_T, p_T), a_T
        else:
            return r_p_to_frames(r_T, p_T), None

    def addnoise(
        self,
        f_s: torch.Tensor,              # [*, L, 4, 4]
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,                # [*]
        s: torch.Tensor = None,         # [*]
        tor_s: torch.Tensor = None,     # [*, L, 4, 2]
        prior_mask: torch.Tensor = None,    # [*, L]
    ):
        r_s, p_s = frames_to_r_p(f_s)
        r_t, _ = self.rot_trans.addnoise(r_s, frame_gen_mask, t, s)
        p_t, _ = self.pos_trans.addnoise(p_s, frame_gen_mask, t, s)
        f_t = r_p_to_frames(r_t, p_t)
        if prior_mask is not None:
            f_prior, a_prior = self.prior(t.shape, seq_len=prior_mask.shape[-1], dtype=f_t.dtype, device=f_t.device)
            f_t = torch.where(
                prior_mask[..., None, None] > 0, f_prior, f_t
            )

        if tor_s is not None:
            assert self.chi_trans is not None
            a_s = sin_cos_to_angles(tor_s)
            a_t = self.chi_trans.addnoise(a_s, frame_gen_mask, t, s)
            if prior_mask is not None:
                a_t = torch.where(
                    prior_mask[..., None] > 0, a_prior, a_t
                )
            tor_t = angles_to_sin_cos(a_t)
            return f_t, tor_t
        else:
            return f_t, None

    def denoise(
        self,
        f_t: torch.Tensor,              # [*, L, 4, 4]
        fh_0: torch.Tensor,
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,                # [*]
        s: torch.Tensor,          # [*]
        tor_t: torch.Tensor = None,
        torh_0: torch.Tensor = None,
        res1_indices: list = [],
        res2_indices: list = [],
        cutoff_list: list = [],
        eta: float = 0.01,
        g_decay: float = 5,
        decay_radius: int = 0,
        decay_lambda: float = 0.1
    ):
        r_t, p_t = frames_to_r_p(f_t)
        rh_0, ph_0 = frames_to_r_p(fh_0)
        r_s, _ = self.rot_trans.denoise(r_t, rh_0, frame_gen_mask, t, s)
        p_s, _ = self.pos_trans.denoise(p_t, ph_0, frame_gen_mask, t, s, res1_indices,res2_indices,cutoff_list, eta, g_decay, decay_radius, decay_lambda)
        f_s = r_p_to_frames(r_s, p_s)
        if tor_t is not None:
            assert self.chi_trans is not None
            a_t = sin_cos_to_angles(tor_t)
            ah_0 = sin_cos_to_angles(torh_0)
            a_s = self.chi_trans.denoise(a_t, ah_0, frame_gen_mask, t, s)
            tor_s = angles_to_sin_cos(a_s)
            return f_s, tor_s
        else:
            return f_s, None

    def langevin(
        self,
        f_t: torch.Tensor,              # [*, L, 4, 4]
        fh_0: torch.Tensor,
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,                # [*]
        sigma_l_r: torch.Tensor = torch.tensor(0.01),
        sigma_l_p: torch.Tensor = torch.tensor(0.05)
    ):
        r_t, p_t = frames_to_r_p(f_t)
        rh_0, ph_0 = frames_to_r_p(fh_0)
        
        # calculate the score function in position and rotation space
        sigma_t_sqs = self.rot_trans.sigma_sq(t)
        score_r = self.rot_trans.score(r_t, rh_0, sigma_t_sqs.sqrt())
        score_p = self.pos_trans.score(p_t, ph_0,t)
        
        # langevin steps in rotation space
        delta_so3 = self.rot_trans.random_so3(
            sigmas=(2 * sigma_l_r).sqrt(),
            size=r_t.shape[-3],
            igso3_obj=self.rot_trans.igso3,
            rw_approx_thres=self.rot_trans.rw_approx_thres
        )   # * L 3
        
        update_vec = score_r * sigma_l_r[..., None, None]
    
        noisy_update_vec = update_vec + delta_so3
        update_mat = self.rot_trans.Exp(noisy_update_vec)

        if self.rot_trans.left:
            r_s = update_mat @ r_t
        else:
            r_s = r_t @ update_mat
            
        if frame_gen_mask is not None:
            r_s = torch.where(
                frame_gen_mask[..., None, None] > 0.,
                r_s, r_t
            )
            
        # do langevin dynamics in position space
        z = np.random.randn(*p_t.shape)
        z = torch.tensor(z, dtype=p_t.dtype, device=p_t.device)
        p_s = p_t + sigma_l_p * score_p + (2 * sigma_l_p).sqrt() * z
        
        if frame_gen_mask is not None:
            p_s = torch.where(
                frame_gen_mask[..., None] > 0,
                p_s,
                p_t
            )
        return r_p_to_frames(r_s, p_s)