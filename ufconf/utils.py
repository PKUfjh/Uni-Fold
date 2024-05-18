import torch
import numpy as np
from numpy import ndarray
from unifold.modules.frame import Frame, Rotation
import torch.nn.functional as F
from unicore.utils import tensor_tree_map
from unifold.data.protein import Protein, to_pdb, from_feature, from_prediction
from ufconf.diffold import Diffold
from ufconf.diffusion import so3
from ufconf.config import model_config
from ufconf.diffusion.diffuser import frames_to_r_p, r_p_to_frames
from unifold.data.lmdb_dataset import LMDBDataset
from unifold.dataset import load_and_process, process
import json
import os
import pickle
import gzip
from unifold.colab import make_input_features
import unifold.data.residue_constants as rc
from ufconf.dataset import load_pdb_feat, load_cif_feat
from unicore.data.data_utils import numpy_seed
from ufconf.diffuse_ops import rbf_kernel,make_noisy_quats
from ufconf.diffusion.diffuser import angles_to_sin_cos, sin_cos_to_angles

from absl import logging
logging.set_verbosity("info")

n_ca_c_trans = torch.tensor(
    [[-0.5250, 1.3630, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [1.5260, -0.0000, -0.0000]],
    dtype=torch.float,
)


def remove_center(*args, mask, eps=1e-12):
    inputs = [Frame.from_tensor_4x4(f) for f in args]
    ref_centers = [(f.get_trans() * mask[..., None]).sum(dim=-2)
                   for f in inputs]
    ref_centers = [ref_center / (mask[..., None].sum(dim=-2) + eps)
                   for ref_center in ref_centers]

    outputs = [Frame(inputs[index].get_rots(), inputs[index].get_trans(
    ) - ref_centers[index]) for index in range(len(inputs))]
    return (o.to_tensor_4x4() for o in outputs)


def chain_feat_map(raw_feats):
    chain_idx_map_tuple = raw_feats.pop('pdb_idx')
    chains = []
    for item in chain_idx_map_tuple:
        if item[0] not in chains:
            chains.append(item[0])
    chain_idx_map = {k: [] for k in chains}
    save_maps = {}
    for tup in chain_idx_map_tuple:
        chain, idx, list_idx = tup
        chain_idx_map[chain].append(list_idx)

    global_index = 0
    for c, idx in chain_idx_map.items():
        c_feats = {}
        global_idx = [int(i) + global_index for i in idx]

        for f, v in raw_feats.items():
            c_feats[f] = v[global_idx]
        save_maps[c] = c_feats
        global_index += len(global_idx)
    return save_maps


def make_mask(seq_len: int, gen_region: str,):
    if gen_region.startswith("+"):
        gen_region = gen_region[1:]
        mask = np.zeros((seq_len,))
        for l in gen_region.strip().split(";"):
            s, e = l.strip().split(":")
            mask[int(s):int(e)] = 1.
        return mask
    else:
        gen_region = gen_region[1:]
        mask = np.ones((seq_len,))
        for l in gen_region.strip().split(";"):
            s, e = l.strip().split(":")
            mask[int(s):int(e)] = 0.
        return mask


def to_numpy(x: torch.Tensor, reduce_batch_dim: bool = False):
    if reduce_batch_dim:
        x = x.squeeze(0)
    if x.dtype in (torch.float, torch.bfloat16, torch.float16):
        x = x.detach().cpu().float().numpy()
    elif x.dtype in (torch.long, torch.int, torch.int64):
        x = x.detach().cpu().long().numpy()
    else:
        raise ValueError(f"unknown dtype {x.dtype}")
    return x


def atom37_to_backb_frames(protein, eps):
    if "all_atom_positions" not in protein:
        return protein

    aatype = protein["aatype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]
    batch_dims = len(aatype.shape[:-1])

    gt_frames = Frame.from_3_points(
        p_neg_x_axis=all_atom_positions[..., 2, :],
        origin=all_atom_positions[..., 1, :],
        p_xy_plane=all_atom_positions[..., 0, :],
        eps=eps,
    )

    rots = torch.eye(3, dtype=all_atom_positions.dtype,
                     device=all_atom_positions.device)
    rots = torch.tile(rots, (1,) * (batch_dims + 2))
    rots[..., 0, 0] = -1.
    rots[..., 2, 2] = -1.
    rots = Rotation(mat=rots)
    gt_frames = gt_frames.compose(Frame(rots, None))

    gt_exists = torch.min(all_atom_mask[..., :3], dim=-1, keepdim=False)[0]

    gt_frames_tensor = gt_frames.to_tensor_4x4()

    protein.update({
        "backb_frames": gt_frames_tensor,
        "backb_frame_mask": gt_exists,
    })

    return protein


def compute_relative_positions(
    res_id: torch.Tensor,
    chain_id: torch.Tensor,
    cutoff: int,
):
    different_chain_symbol = -(cutoff + 1)
    relpos = res_id[..., None] - res_id[..., None, :]
    relpos = relpos.clamp(-cutoff, cutoff)

    different_chain = (chain_id[..., None] != chain_id[..., None, :])
    relpos[different_chain] = different_chain_symbol

    return relpos


def compute_atomic_positions(
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
):
    frames = Frame.from_tensor_4x4(frames)
    n_ca_c = frames[..., None].apply(n_ca_c_trans.to(frames.device))
    relpos = compute_relative_positions(residue_index, chain_id, cutoff=2)
    is_next_res = (relpos == -1).float()    # [*, L, L]
    next_n_ca_c = torch.einsum(
        "...jad,...ij->...iad", n_ca_c, is_next_res
    )   # [*, L, na=3, d=3]
    next_frame_exist = torch.einsum(
        "...j,...ij->...i", seq_mask, is_next_res
    )   # [*, L]
    oxygen_frames = Frame.from_3_points(
        n_ca_c[..., 1, :],
        n_ca_c[..., 2, :],
        next_n_ca_c[..., 0, :],
    )
    oxygen_coord = oxygen_frames.apply(
        torch.tensor(
            [0.627, -1.062, 0.000],
            dtype=n_ca_c.dtype, device=n_ca_c.device
        )
    )[..., None, :]
    n_ca_c_o = torch.cat((
        n_ca_c,
        oxygen_coord.new_zeros(oxygen_coord.shape),
        oxygen_coord
    ), dim=-2)
    atom_pos = F.pad(n_ca_c_o, (0, 0, 0, 32))
    atom_mask = torch.stack((
        seq_mask, seq_mask, seq_mask,
        seq_mask.new_zeros(seq_mask.shape),
        seq_mask
    ), dim=-1)
    atom_mask = F.pad(atom_mask, (0, 32))
    return atom_pos, atom_mask


def to_pdb_string(
    aatype: torch.Tensor,
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    b_factor: torch.Tensor = None,
    model_id: int = 1,
):
    # n_ca_c = n_ca_c_trans.to(frames.device)
    # aatype = aalogits[..., :20].argmax(dim=-1)  # L
    atom_pos, atom_mask = compute_atomic_positions(
        frames, seq_mask, residue_index, chain_id)
    if b_factor is None:
        b_factor = atom_mask
    else:
        assert b_factor.shape == atom_mask.shape

    prot_dict = {
        "atom_positions": atom_pos,
        "aatype": aatype,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
        "chain_index": chain_id - 1,
        "b_factors": b_factor
    }
    has_batch_dim = (len(aatype.shape) == 2)
    prot_dict = tensor_tree_map(
        lambda x: to_numpy(x, reduce_batch_dim=has_batch_dim),
        prot_dict
    )

    prot = Protein(**prot_dict)
    ret = to_pdb(prot, model_id=model_id)
    return ret


def save_pdb(
    path: str,
    aatype: torch.Tensor,
    frames: torch.Tensor,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    b_factor: torch.Tensor = None,
    model_id: int = 1,
):
    pdb_string = to_pdb_string(
        aatype,
        frames,
        seq_mask,
        residue_index,
        chain_id,
        b_factor,
        model_id,
    )

    with open(path, 'w') as f:
        f.write(pdb_string)


def make_output(batch, out=None):
    def to_float(x):
        if x.dtype == torch.bfloat16 or x.dtype == torch.half:
            return x.float()
        else:
            return x
    # batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
    batch = tensor_tree_map(to_float, batch)
    # out = tensor_tree_map(lambda t: t[0, ...], out)
    batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
    if out is not None:
        b_factor = out["plddt"][..., None].tile(37).squeeze()
        b_factor = np.array(b_factor.cpu())
        out = tensor_tree_map(to_float, out)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        # cur_protein = from_prediction(
        #     features=batch, result=out, b_factors=None
        # )
        cur_protein = from_prediction(
            features=batch, result=out, b_factors=b_factor
        )
        return cur_protein
    else:
        cur_protein = from_feature(features=batch)
        return cur_protein


def kabsch(P: ndarray, Q: ndarray) -> ndarray:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U: ndarray = np.dot(V, W)

    return U


def kabsch_rotate(P: ndarray, Q: ndarray) -> ndarray:
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P, U


def interpolate_positions(pos1: torch.Tensor, pos2: torch.Tensor, fraction: float) -> torch.Tensor:
    return (1 - fraction) * pos1 + fraction * pos2


def interpolate_rotations(rot1: torch.Tensor, rot2: torch.Tensor, fraction: float) -> torch.Tensor:
    return rot1.cpu() @ so3.Exp(fraction * so3.Log(rot1.cpu().transpose(-1, -2) @ rot2.cpu()))


def interpolate_conf(ft_0: torch.Tensor, ft_1: torch.Tensor, num_steps: int) -> list:
    interpolate_ft_list = []
    # ft_1 = align_frame(ft_1,ft_0, frame_gen_mask)
    rt_0, pt_0 = frames_to_r_p(ft_0)
    rt_1, pt_1 = frames_to_r_p(ft_1)

    for fraction in [i / (num_steps) for i in range(num_steps + 1)]:
        interpolate_p = interpolate_positions(pt_0, pt_1, fraction)
        interpolate_r = interpolate_rotations(rt_0, rt_1, fraction)
        interpolate_r = torch.tensor(interpolate_r, device=ft_0.device)
        interpolate_ft = r_p_to_frames(interpolate_r, interpolate_p)
        interpolate_ft_list.append(interpolate_ft)
    return interpolate_ft_list


def config_and_model(args):
    config = model_config(args.model, train=False)
    print("config keys", config.keys())
    model = Diffold(config)

    if args.checkpoint is not None:
        logging.info("start to load params {}".format(args.checkpoint))
        state_dict = torch.load(args.checkpoint)
        print("state keys", state_dict.keys())
        if "ema" in state_dict:
            logging.info("ema model exist. using ema.")
            state_dict = state_dict["ema"]["params"]
        else:
            logging.info("no ema model exist. using original params.")
            state_dict = state_dict["model"]
        state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        # print("state_dict",state_dict)
        model.load_state_dict(state_dict, strict = False)
    else:
        logging.warning("*** UNRELIABLE RESULTS!!! ***")
        logging.warning(
            "checkpoint not provided. running model with random parameters.")

    model = model.to(args.device)
    model.eval()
    if args.bf16:
        model.bfloat16()

    return config, model


def compute_theta_translation(f_t: torch.Tensor, f_0: torch.Tensor, gamma: float, gen_frame_mask=None):
    """
    Inputs: 
    * f_t: (..., 4, 4) tensor. current frame
    * f_0: (..., 4, 4) tensor. the reference frame
    * gamma: the variance of the forward process
    * gen_frame_mask: the mask used to define the generated regions on the sequence
    Outputs: 
    * theta: the rotation angle
    * translation: the translation vector
    """
    if gen_frame_mask is not None:
        gen_frame_mask = gen_frame_mask.squeeze()
        bool_array = to_numpy(gen_frame_mask).astype(bool)
        f_0 = f_0[bool_array, :, :]
        f_t = f_t[bool_array, :, :]
    print("new f0 shape", f_0.shape)
    print("new ft shape", f_t.shape)
    r_0, p_0 = torch.split(f_0[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
    r_t, p_t = torch.split(f_t[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]

    theta = so3.theta_and_axis(
        so3.Log((r_0.cpu().transpose(-1, -2) @ r_t.cpu()).numpy()))[0]
    translation = (p_t - gamma.sqrt() * p_0).cpu().numpy()
    return theta, translation


def load_features_from_lmdb(args, config, Job, job_name):
    data_path = args.data_path
    feat_lmdb = LMDBDataset(data_path + "features.lmdb")
    lab_lmdb = LMDBDataset(data_path + "labels.lmdb")
    feat_id_map = json.load(open(data_path + "train_label_to_seq.json"))
    sym_id_map = json.load(open(data_path + "train_mmcif_assembly.json"))
    prot_id = Job["id"]
    print("pdb id", prot_id)

    lids = prot_id
    if type(lids) is str:
        lids = [lids]
    sids = [feat_id_map[l] for l in lids]

    if "is_monomer" not in Job:
        is_monomer = True
    else:
        is_monomer = Job["is_monomer"]
    if not is_monomer:
        symmetry_operations = sym_id_map[job_name]["opers"]
    else:
        symmetry_operations = None

    # preprocess the MSA in the downloaded MSA dataset
    feat, lab = load_and_process(
        config.data,
        "predict",
        batch_idx=0,
        data_idx=int(args.data_idx),
        sequence_ids=sids,
        feature_dir=feat_lmdb,
        msa_feature_dir=data_path + "msa_features",
        template_feature_dir=data_path + "template_features",
        uniprot_msa_feature_dir=data_path + "uniprot_features",
        label_ids=lids,
        label_dir=lab_lmdb,
        symmetry_operations=symmetry_operations
    )

    # print out the number of chains of the protein
    print("chain number", len(lab))
    return feat, lab


def load_features_from_pdb(args, config, Job, dir_feat_name, cif=False):
    pdb_name = Job["pdb"]
    print("pdb name", pdb_name)
    if "symmetry_operations" in Job:
        symmetry_operations = Job["symmetry_operations"]
    else:
        symmetry_operations = None

    # generate initial features from a given pdb file
    if cif:
        pdb_path = os.path.join(args.input_pdbs, pdb_name + ".cif")
        feat = load_cif_feat(pdb_path)
    else:
        pdb_path = os.path.join(args.input_pdbs, pdb_name + ".pdb")
        feat = load_pdb_feat(pdb_path)

    feat = chain_feat_map(feat)

    all_chain_labels = []
    seq_ids = sorted(list(feat.keys()))
    for key in seq_ids:
        print("key", key)
        labels = {
            k: feat[key][k]
            for k in (
                "aatype",
                "all_atom_positions",
                "all_atom_mask"
            )
        }
        all_chain_labels.append(labels)
        labels["resolution"] = np.array([0.])
        pickle.dump(labels, gzip.open(
            f"{dir_feat_name}/{key}.label.pkl.gz", "wb"))

    aatype2resname = {v: k for k, v in rc.restype_order_with_x.items()}

    def map_fn(x): 
        return aatype2resname[x]

    # generate sequences from input features
    seqs = [''.join(list(map(map_fn, feat[i]['aatype']))) for i in seq_ids]
    print("seq_ids", seq_ids)
    print("seqs", seqs)

    seq_list_all = []
    for seq in seqs:
        seq_list = [rc.restype_1to3[res_id] for res_id in seq if res_id != "X"]
        seq_list_all.append(seq_list)
    for seq_list in seq_list_all:
        print("seq_list", seq_list)

    # generate all the MSA for the given sequence
    seqs, seq_ids, feat = make_input_features(
        dir_feat_name,
        seqs,
        seq_ids,
        msa_file_path=args.msa_file_path,
        use_msa=True,
        use_exist_msa=args.use_exist_msa,
        use_templates=False,
        verbose=True,
        min_length=2,
        max_length=2000,
        max_total_length=3000,
        is_monomer=False,
        load_labels=True,
        use_mmseqs_paired_msa=False,
        symmetry_operations=symmetry_operations
    )

    return feat, all_chain_labels


def set_prior(
    features, diffuser, seed, config
):
    frame_mask = features["frame_mask"]
    t = features["diffusion_t"]
    frame_gen_mask = features["frame_gen_mask"]
    
    tor_0 = features["chi_angles_sin_cos"] if config.chi.enabled else None
    f_0 = features["true_frame_tensor"]

    with numpy_seed(seed, 0, key="prior"):
        f_prior, a_prior = diffuser.prior(t.shape, seq_len=frame_mask.shape[-1], dtype=f_0.dtype, device=f_0.device)
        noisy_frames = torch.where(
            frame_mask[..., None, None] > 0, f_prior, f_0
        )

    features["noisy_frames"] = noisy_frames
    if tor_0 is not None:
        a_0 = sin_cos_to_angles(tor_0)
        a_t = torch.where(
                    frame_mask[..., None] > 0, a_prior, a_0
                )
        noisy_torsions = angles_to_sin_cos(a_t)
        noisy_torsions = torch.nan_to_num(noisy_torsions, 0.)
        noisy_torsions = noisy_torsions * features["chi_mask"][..., None]
        features["noisy_chi_sin_cos"] = noisy_torsions
    features = make_noisy_quats(features)

    residue_t = t[..., None]
    # setting motif ts to 0
    residue_t = torch.where(
        frame_gen_mask > 0., residue_t, torch.zeros_like(residue_t),
    )
    # setting unknown frame ts to 1
    residue_t = torch.where(
        frame_mask > 0., residue_t, torch.ones_like(residue_t),
    )

    time_feat = rbf_kernel(residue_t, config.d_time, 0., 1.)
    features["time_feat"] = time_feat
    return features

def recur_print(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return f"{x.shape}_{x.dtype}"
    elif isinstance(x, dict):
        return {k: recur_print(v) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return [recur_print(v) for v in x]
    else:
        raise RuntimeError(x)
