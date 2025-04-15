import torch
import numpy as np
from numpy import ndarray
import time
import copy
import json
import os
import pickle
import gzip
import urllib
import io
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from typing import Optional
from Bio.PDB import PDBParser
import torch.nn.functional as F
from unifold.modules.frame import Frame, Rotation
from unicore.utils import tensor_tree_map
from unifold.data.protein import Protein, to_pdb, from_feature, from_prediction
from ufconf.ufconf import UFConformer
from ufconf.diffusion import so3
from ufconf.config import model_config
from ufconf.diffusion.diffuser import frames_to_r_p, r_p_to_frames
from unifold.colab import make_input_features
import unifold.data.residue_constants as rc
from unicore.data.data_utils import numpy_seed
from ufconf.diffuse_ops import rbf_kernel,make_noisy_quats
from ufconf.diffusion.diffuser import angles_to_sin_cos, sin_cos_to_angles
from unifold.losses.geometry import kabsch_rotation as kabsch
from unifold.dataset import UnifoldDataset
import pandas as pd

from absl import logging
logging.set_verbosity("info")

n_ca_c_trans = torch.tensor(
    [[-0.5250, 1.3630, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [1.5260, -0.0000, -0.0000]],
    dtype=torch.float,
)

def setup_directories(args, job_name, Job, mode = "denoise"):
    """
    Sets up and returns the paths for the main output directory, trajectory output directory, and features directory.

    Args:
        args: Parsed command-line arguments.
        job_name (str): Name of the current job.
        Job (dict): Configuration dictionary for the current job.

    Returns:
        tuple: A tuple containing paths to the output directory, trajectory directory, and features directory.
    """

    # Main output directory, named based on job details
    if mode == "denoise":
        output_dir = os.path.join(args.outputs, job_name, f"tem{Job['initial_t']}_rep{Job['num_replica']}_inf{Job['inf_steps']}")
    elif mode == "langevin":
        output_dir = os.path.join(args.outputs, job_name, f"tem{Job['initial_t']}_rep{Job['num_replica']}_num{Job['num_steps']}_inf{Job['inf_steps']}")
    else:
        raise ValueError("Invalid mode, provide either denoise or langevin.")
    os.makedirs(output_dir, exist_ok=True)

    # Directory for trajectory outputs
    output_traj_dir = os.path.join(output_dir, "traj")
    os.makedirs(output_traj_dir, exist_ok=True)

    # Directory for features
    dir_feat_name = os.path.join(args.outputs, job_name, 'features')
    os.makedirs(dir_feat_name, exist_ok=True)

    return output_dir, output_traj_dir, dir_feat_name

def save_config_json(checkpoint_path, output_dir, Job):
    """
    Saves the job configuration to a JSON file in the output directory.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_dir (str): Directory where the output is to be saved.
        Job (dict): Configuration dictionary for the current job.
    """
    config_data = {
        "timestamp": time.strftime("%Y-%m-%d;%H:%M:%S", time.localtime()),
        "diffusion_t": Job["initial_t"],
        "checkpoint": checkpoint_path
    }

    # Define the file path
    config_file_path = os.path.join(output_dir, "config.json")

    # Writing the configuration data to the JSON file
    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=2)
        
def prepare_features(feat, Job):
    """
    Prepares features based on the job configuration.

    Args:
        feat (dict): Dictionary containing the initial feature data.
        Job (dict): Configuration dictionary for the current job.

    Returns:
        tuple: A tuple containing the modified feature dictionary and the residue list.
    """

    # Deep copy of the feature dictionary to avoid modifying the original
    featd = copy.deepcopy(feat)
    featd["diffusion_t"] = torch.tensor([Job["initial_t"]])
    featd["true_frame_tensor"] = featd["true_frame_tensor"].squeeze()

    # Handling residue indices if specified in Job configuration
    residue_list = None
    if "residue_idx" in Job:
        residue_list = []
        if isinstance(Job["residue_idx"], str):
            for part in Job["residue_idx"].split("/"):
                residue_idx = part.split("-")
                residue_indices = list(range(int(residue_idx[0]), int(residue_idx[1]) + 1))
                residue_list.extend(residue_indices)
        elif isinstance(Job["residue_idx"], list):
            residue_list = [int(i) for i in Job["residue_idx"]]

    # Generating mask for residues to be generated
    is_gen = torch.zeros_like(feat["frame_mask"])
    if residue_list is not None:
        is_gen[:, residue_list] = 1.
    else:
        is_gen = torch.ones_like(feat["frame_mask"])
    featd["frame_gen_mask"] = feat["frame_mask"] * is_gen

    return featd, residue_list
        
def prepare_batch(featd, lab, args):
    """
    Prepares the batch for processing by collating feature data and labels and transferring to the specified device.

    Args:
        featd (dict): Modified feature dictionary for a specific job and replica.
        lab (dict): Labels corresponding to the feature data.
        args: Parsed command-line arguments containing device information.

    Returns:
        dict: The batch data ready for processing, transferred to the specified device.
    """

    # Collate the features into a batch
    batch = UnifoldDataset.collater([featd])

    # Transfer batch data to the specified device (e.g., GPU)
    batch_device = {k: torch.as_tensor(v, device=args.device) for k, v in batch.items()}

    if lab:
        # Include additional batch data from labels
        chain_id_list = []
        for chain_index in range(len(lab)):
            chain_id = torch.tensor([chain_index + 1] * lab[chain_index]["aatype"].shape[0], device=args.device)
            chain_id_list.append(chain_id)
        batch_device["chain_id"] = torch.cat(chain_id_list, dim=0)
    else:
        chain_id_list = []
        chain_id = torch.tensor([1] * featd["aatype"].squeeze().shape[0], device=args.device)
        chain_id_list.append(chain_id)
        batch_device["chain_id"] = torch.cat(chain_id_list, dim=0)
        
    return batch_device

def extract_index_lists(csv_file, top_n=20):
    """Extract residue index pairs and distances from a CSV file."""
    df = pd.read_csv(csv_file)
    df = df.head(top_n)

    res1_indices = df["Residue1_Index"].tolist()
    res2_indices = df["Residue2_Index"].tolist()

    return res1_indices, res2_indices

# handle pdb format files
def get_pdb(filename):
    if not os.path.exists(filename):
        print('pdb not exist, download ... ')
        pdb_id = filename.split('/')[-1].split('.')[0]
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, filename)
        print(f'finish download {pdb_id}')
    return filename

def read_pdb(path, line_list=None):
    with open(path, 'r') as f:
        if line_list:
            return  f.readlines()
        else: return f.read() 

def load_pdb_feat(filename):
    pdb_file = get_pdb(filename)
    return from_pdb_string(read_pdb(pdb_file))


# add heteratom ?
def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None, No_model=0) -> dict:
    """Takes a PDB string and constructs a Protein object.
    #### NOTICE: the difference between `unifold/data/protein.py` is that here we return dict type.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        if No_model == 0:
            logger.warn(f'found {len(models)} models in the pdb, and using default model {No_model}, pease make sure you want No. `{No_model}` model in the pdb files.')
    model = models[No_model]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    res_list_idx = []
    chain_ids = []
    b_factors = []
    res_shortnames = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res_idx, res in enumerate(chain):
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue " +
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = rc.restype_3to1.get(res.resname, "X")
            res_shortnames.append(res_shortname)
            restype_idx = rc.restype_order.get(
                res_shortname, rc.restype_num
            )
            pos = np.zeros((rc.atom_type_num, 3))
            mask = np.zeros((rc.atom_type_num,))
            res_b_factors = np.zeros((rc.atom_type_num,))
            for atom in res:
                if atom.name not in rc.atom_types:
                    continue
                pos[rc.atom_order[atom.name]] = atom.coord
                mask[rc.atom_order[atom.name]] = 1.0
                res_b_factors[rc.atom_order[atom.name]] = atom.bfactor
            # if np.sum(mask) < 0.5:    # keep
            #     # If no known atom positions are reported for the residue then skip it.
            #     continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            res_list_idx.append(res_idx)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    chain_index_map = [(c, r, i) for c,r,i in zip(chain_ids, residue_index, res_list_idx)]

    return {
        'all_atom_positions' : np.array(atom_positions),
        'all_atom_mask' : np.array(atom_mask),
        'aatype' : np.array(aatype),
        'residue_index' : np.array(residue_index),
        'chain_index' : chain_index,
        'pdb_idx' : chain_index_map,
        'b_factors' : np.array(b_factors),
        'res_seq' : np.array(res_shortnames)
    }

# handle cif format file
def get_cif(filename):
    if not os.path.exists(filename):
        print('CIF file does not exist, downloading...')
        pdb_id = filename.split('/')[-1].split('.')[0]
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        urllib.request.urlretrieve(url, filename)
        print(f'Finished downloading {pdb_id}')
    return filename

def load_cif_feat(filename,chain_id: Optional[str] = None):
    cif_file = get_cif(filename)
    mmcif_dict = MMCIF2Dict(cif_file)
    
    parser = MMCIFParser()
    structure = parser.get_structure("none", cif_file)
    model = next(structure.get_models())  # Assuming interest in the first model
    
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    res_list_idx = []
    chain_ids = []
    b_factors = []
    res_shortnames = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res_idx, res in enumerate(chain):
            # if res.id[2] != " ":
            #     raise ValueError(
            #         f"PDB contains an insertion code at chain {chain.id} and residue " +
            #         f"index {res.id[1]}. These are not supported."
            #     )
            res_shortname = rc.restype_3to1.get(res.resname, "X")
            res_shortnames.append(res_shortname)
            restype_idx = rc.restype_order.get(
                res_shortname, rc.restype_num
            )
            pos = np.zeros((rc.atom_type_num, 3))
            mask = np.zeros((rc.atom_type_num,))
            res_b_factors = np.zeros((rc.atom_type_num,))
            for atom in res:
                if atom.name not in rc.atom_types:
                    continue
                pos[rc.atom_order[atom.name]] = atom.coord
                mask[rc.atom_order[atom.name]] = 1.0
                res_b_factors[rc.atom_order[atom.name]] = atom.bfactor
            # if np.sum(mask) < 0.5:    # keep
            #     # If no known atom positions are reported for the residue then skip it.
            #     continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            res_list_idx.append(res_idx)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    chain_index_map = [(c, r, i) for c,r,i in zip(chain_ids, residue_index, res_list_idx)]

    return {
        'all_atom_positions' : np.array(atom_positions),
        'all_atom_mask' : np.array(atom_mask),
        'aatype' : np.array(aatype),
        'residue_index' : np.array(residue_index),
        'chain_index' : chain_index,
        'pdb_idx' : chain_index_map,
        'b_factors' : np.array(b_factors),
        'res_seq' : np.array(res_shortnames)
    }

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
    P = P @ U
    # P = np.dot(P, U)
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
    model = UFConformer(config)

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

def load_features_from_pdb(args, Job, dir_feat_name, cif=False):
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

from Bio import SeqIO
import re

def load_features_from_fasta(args, Job, dir_feat_name=None):
    fasta_name = Job["fasta"]
    if "symmetry_operations" in Job:
        symmetry_operations = Job["symmetry_operations"]
    else:
        symmetry_operations = None

    # Extract the sequence from the input FASTA file
    fasta_path = os.path.join(args.input_pdbs, fasta_name + ".fasta")
    
    # Initialize lists for sequence IDs and sequences
    seq_ids = []
    seqs = []

    # Parse the FASTA file
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Use regex to find the chain identifier after "Chain" in the description, if present
        match = re.search(r"Chain (\w)", record.description)
        
        # If a chain identifier is found, append it; otherwise, set chain ID to 'A'
        if match:
            seq_ids.append(match.group(1))  # Add the chain identifier to the list
        else:
            # Default chain ID as 'A' if not found
            seq_ids.append("A")
        
        seqs.append(str(record.seq))

    aatype = []
    residue_index = []
    res_list_idx = []
    chain_ids = []
    
    # Process the sequences and residues
    for seq_idx in range(len(seqs)):
        seq = seqs[seq_idx]
        chain_id = seq_ids[seq_idx]
        for res_idx in range(len(seq)):
            res_shortname = seq[res_idx]
            restype_idx = rc.restype_order.get(
                res_shortname, rc.restype_num
            )
            aatype.append(restype_idx)
            residue_index.append(res_idx)
            res_list_idx.append(res_idx)
            chain_ids.append(chain_id)

    chain_index_map = [(c, r, i) for c,r,i in zip(chain_ids, residue_index, res_list_idx)]
    
    # Create feature dictionary
    feat = {
        'aatype': np.array(aatype),
        'residue_index': np.array(residue_index),
        'pdb_idx': chain_index_map
    }

    # Apply chain feature mapping (if necessary)
    feat = chain_feat_map(feat)

    # Initialize list to store all chain labels
    all_chain_labels = []
    seq_ids = sorted(list(feat.keys()))
    
    for key in seq_ids:
        print("key", key)
        labels = {
            "aatype": feat[key]["aatype"]
        }
        all_chain_labels.append(labels)
        labels["resolution"] = np.array([0.])
        
        # Save the labels to a file
        pickle.dump(labels, gzip.open(
            f"{dir_feat_name}/{key}.label.pkl.gz", "wb"))
    
    # Generate all the MSA for the given sequence (if applicable)
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
        load_labels=False,
        use_mmseqs_paired_msa=False,
        symmetry_operations=symmetry_operations
    )

    return feat, all_chain_labels