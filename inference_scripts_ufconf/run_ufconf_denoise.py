# from unifold.inference import *
import argparse
import os
import json
import torch
import time
from typing import *
import tqdm
import pickle

from absl import logging
logging.set_verbosity("info")
from ufconf.diffusion.diffuser import Diffuser
from ufconf.diffuse_ops import diffuse_inputs, make_noisy_quats, rbf_kernel
from unifold.data.protein import to_pdb
from unifold.dataset import process
import copy
import random
import ufconf.utils as utils

max_retries = 3  # Maximum number of retries
retry_delay = 3  # Delay between retries in seconds

def process_replica(args, model, batch, diffuser, config, Job, output_traj_dir, job_name, replica, res1_indices,res2_indices, cutoff_list, eta, g_decay, decay_radius, decay_lambda):
    batch_constants = {
        key: batch[key].squeeze() for key in ("seq_mask", "residue_index", "chain_id")
    }
    rep_name = f"{job_name}"
    time_stamps = torch.linspace(Job["initial_t"], 0., Job["inf_steps"] + 1).float().to(args.device)
    # output the ground truth structure including side chain atoms
    with open(os.path.join(output_traj_dir, f"f0_{rep_name}.pdb"), "w") as f:
        batch_squeeze = copy.deepcopy(batch)
        batch_squeeze["asym_id"] = batch["asym_id"].squeeze()
        batch_squeeze["aatype"] = batch["aatype"].squeeze()
        batch_squeeze["all_atom_positions"] = batch["all_atom_positions"].squeeze()
        batch_squeeze["residue_index"] = batch["residue_index"].squeeze()
        batch_squeeze["all_atom_mask"] = batch["all_atom_mask"].squeeze()
        f.write(to_pdb(utils.make_output(batch_squeeze)))
    
    iter_t = (tqdm.tqdm(range(Job["inf_steps"]), total=Job["inf_steps"]) \
        if Job["inf_steps"] >= 10 else range(Job["inf_steps"]))
    s_t, f_t = batch["aatype"].squeeze(), batch["noisy_frames"].squeeze()
    
    if Job["save_trajectory"]:
        with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
            f.write(utils.to_pdb_string(
                s_t, f_t, **batch_constants, model_id=0,
            ))

    with open(os.path.join(output_traj_dir, f"ft_{rep_name}.pdb"), "a") as f:
        f.write(utils.to_pdb_string(
            s_t, f_t, **batch_constants, model_id=replica + 1
        ))
    for i in iter_t:
        # denoise step
        t, s = time_stamps[i], time_stamps[i + 1]
        assert batch["diffusion_t"] == t
        with torch.no_grad():
            out = model(batch)
            if config.diffusion.chi.enabled:
                pred_chi_angles_sin_cos = out["sm"]["angles"][-1,0,:,3:,:]
            ret_frames = out["pred_frame_tensor"]
        if config.diffusion.chi.enabled:
            batch["chi_angles_sin_cos"] = batch["chi_angles_sin_cos"] *  batch["chi_mask"][..., None]
            pred_chi_angles_sin_cos = pred_chi_angles_sin_cos * batch["chi_mask"][..., None]
            tor_t = batch["noisy_chi_sin_cos"].squeeze()
            tor_0 =  batch["chi_angles_sin_cos"].squeeze()
            torh_0 = pred_chi_angles_sin_cos.squeeze()
        else:
            tor_t = tor_0 = torh_0 = None
        sh_0, fh_0 = batch["aatype"].squeeze(), ret_frames.squeeze()
        f_t = batch["noisy_frames"].squeeze()
        with torch.no_grad():
            f_s, tor_s  = diffuser.denoise(
                f_t, fh_0,
                batch["frame_gen_mask"],
                t=t, s=s,
                tor_t=tor_t, torh_0=torh_0,
                res1_indices = res1_indices,
                res2_indices = res2_indices,
                cutoff_list = cutoff_list,
                eta = eta,
                g_decay = g_decay,
                decay_radius = decay_radius, 
                decay_lambda = decay_lambda
            )
        batch["noisy_frames"] = f_s
        if tor_s is not None:
            tor_s = tor_s * batch["chi_mask"][...,None]
            batch["noisy_chi_sin_cos"] = tor_s
        batch["diffusion_t"] = torch.tensor([[s]], device=args.device)
        s_t, f_t = batch["aatype"].squeeze(), batch["noisy_frames"].squeeze()
    
        b_factor = out["plddt"][..., None].tile(37).squeeze()
        if Job["save_trajectory"]:
            # output the predicted structures during reverse process into pdb files
            with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    sh_0, fh_0, **batch_constants, model_id=i + 1, b_factor=b_factor
                ))
            # output the noisy structures during reverse process into pdb files
            with open(os.path.join(output_traj_dir, f"ft_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
                f.write(utils.to_pdb_string(
                    s_t, f_t, **batch_constants, model_id=i + 1,
                ))
    
        # update features
        batch = make_noisy_quats(batch)
        # print("batch diffusion t",batch["diffusion_t"])
        residue_t = batch["diffusion_t"][..., None]
        # setting motif ts to 0
        residue_t = torch.where(
            batch["frame_gen_mask"] > 0., residue_t, torch.zeros_like(residue_t),
        )
        # setting unknown frame ts to 1
        residue_t = torch.where(
            batch["frame_mask"] > 0., residue_t, torch.ones_like(residue_t),
        )
        time_feat = rbf_kernel(residue_t, config.diffusion.d_time, 0., 1.)
        batch["time_feat"] = time_feat

    # last step of the prediction
    t = time_stamps[Job["inf_steps"]]
    assert batch["diffusion_t"] == t
    with torch.no_grad():
        out = model(batch)
        ret_frames = out["pred_frame_tensor"]
    sh_0, fh_0 = batch["aatype"].squeeze(), ret_frames.squeeze()
    f_t = batch["noisy_frames"].squeeze()
    b_factor = out["plddt"][..., None].tile(37).squeeze()
    if Job["save_trajectory"]:
        # output the predicted structures during reverse process into pdb files
        with open(os.path.join(output_traj_dir, f"f0h_inf_{rep_name}_rep{replica}.pdb"), "a") as f:
            f.write(utils.to_pdb_string(
                sh_0, fh_0, **batch_constants, model_id=Job["inf_steps"] + 1, b_factor=b_factor
            ))
    # output the predicted structure including side chain atoms
    with open(os.path.join(output_traj_dir, f"f0h_{rep_name}.pdb"), "a") as f:
        batch_squeeze = copy.deepcopy(batch)
        out_squeeze = copy.deepcopy(out)
        batch_squeeze["asym_id"] = batch["asym_id"].squeeze()
        batch_squeeze["aatype"] = batch["aatype"].squeeze()
        batch_squeeze["residue_index"] = batch["residue_index"].squeeze()
        out_squeeze["final_atom_positions"] = out["final_atom_positions"].squeeze()
        out_squeeze["final_atom_mask"] = out["final_atom_mask"].squeeze()
        prot = utils.make_output(batch_squeeze, out_squeeze)
        f.write(to_pdb(prot, model_id=replica + 1))

def main(args):
    """
    Main function to run denoising inference for protein structures using ufconf model.

    This function takes a configuration file specifying various jobs, loads the ufconf model from a checkpoint,
    and processes each job according to the given configurations. 
    
    It supports:
    1. processing from a pre-existing PDB file and generating  MSA (Multiple Sequence Alignment) on the fly.
    2. using existing downloaded PDB dataset and MSA dataset to run inference.

    Parameters:
    - args: An argparse.Namespace object containing the following attributes:
        - tasks (str): Path to a JSON file specifying the tasks for protein structure inference.
        - input_pdbs (str): Directory of input reference PDB files.
        - outputs (str): Directory to save the output results.
        - checkpoint (str): Path to the model checkpoint file.
        - device (str): The device to run the model on (e.g., 'cuda:0').
        - chunk_size (str): Chunk size for processing.
        - model (str): Name of the ufconf model to use.
        - data_idx (str): Index to specify the random seed for MSA.
        - from_pdb (bool): Flag to indicate whether to run inference with on-the-fly generated MSA.
        - bf16 (bool): Flag to indicate the use of bfloat16 precision.

    The function processes each job by loading features and labels, performing protein structure inference, and saving the results.
    It supports generating structures with different parameters such as diffusion time, number of replicas, and number of steps, 
    and saves trajectories and final structures in the specified output directory.
    """
    with open(args.tasks, "r") as f:
        job_configs = json.load(f)

    # load the models from checkpoint
    checkpoint_path = args.checkpoint
    tic = time.perf_counter()
    config, model = utils.config_and_model(args)
    toc = time.perf_counter() - tic
    logging.info(f"model initialized in {toc:.2f} seconds.")
    
    diffuser = Diffuser(config.diffusion)
    gpu_diffuser = Diffuser(config.diffusion).to(args.device)
        
    config.data.predict.supervised = True
    config.globals.chunk_size = int(args.chunk_size)
    print("config chunk size", config.globals.chunk_size)
    job_name_list = list(job_configs.keys())
    Job_list = [job_configs[job_name] for job_name in job_name_list]

    # Iterate over each job
    for job_name, Job in zip(job_name_list, Job_list):
        output_dir, output_traj_dir, dir_feat_name = utils.setup_directories(args, job_name, Job, mode = "denoise")
        utils.save_config_json(checkpoint_path, output_dir, Job)
        # If use `from_cif` flag, then generate MSA based on the input cif file.
        if args.from_cif:
            print("loading features from cif...")
            for attempt in range(max_retries):
                try:
                    # Attempt to load the features
                    feat_raw, all_chain_labels = utils.load_features_from_pdb(args, Job, dir_feat_name, cif=True)
                    break  # If successful, exit the loop
                except Exception as e:  # Catching a general exception
                    if attempt < max_retries - 1:
                        # If not the last attempt, wait and then retry
                        print(f"EOFError encountered. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        # If it was the last attempt, re-raise the exception
                        print("Maximum retries reached. Aborting.")
                        raise
        elif args.from_pdb:
            print("loading features from pdb...")
            for attempt in range(max_retries):
                try:
                    # Attempt to load the features
                    feat_raw, all_chain_labels = utils.load_features_from_pdb(args, Job, dir_feat_name, cif=False)
                    break  # If successful, exit the loop
                except Exception as e:  # Catching a general exception
                    if attempt < max_retries - 1:
                        # If not the last attempt, wait and then retry
                        print(f"EOFError encountered. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        # If it was the last attempt, re-raise the exception
                        print("Maximum retries reached. Aborting.")
                        raise
        elif args.from_fasta:
            print("loading features from fasta...")
            for attempt in range(max_retries):
                try:
                    # Attempt to load the features
                    feat_raw, all_chain_labels = utils.load_features_from_fasta(args, Job, dir_feat_name)
                    # Save it as a .pkl file
                    with open(os.path.join(dir_feat_name,'features.pkl'), 'wb') as f:
                        pickle.dump(feat_raw, f)
                    break  # If successful, exit the loop
                except Exception as e:  # Catching a general exception
                    if attempt < max_retries - 1:
                        # If not the last attempt, wait and then retry
                        print(f"EOFError encountered. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        # If it was the last attempt, re-raise the exception
                        print("Maximum retries reached. Aborting.")
                        raise
        else:
            print(f"Please provide valid flag: 'from_fasta, ''from_pdb' or 'from_cif'.")
            
        if args.use_guidance:
            guide_csv = os.path.join(args.input_pdbs, f'{Job["guidance_csv"]}.csv')
            res1_indices, res2_indices, cutoff_list = utils.extract_index_lists(guide_csv,200)
        else:
            res1_indices = []
            res2_indices = []
            cutoff_list = []
            
        for replica in tqdm.tqdm(range(Job["num_replica"]), total=Job["num_replica"]) \
            if Job["num_replica"] >= 10 else range(Job["num_replica"]):
            logging.info(f"running replica {replica+1}/{Job['num_replica']}...")
            replica = replica + args.start * Job["num_replica"]
            my_seed = random.randint(0, 1000000)  # Generate a random seed
            # preprocess all the MSA, make random deletion to the MSAs controlled by the random seed defined in `data_idx`
            feat, lab = process(
                config.data,
                mode="predict",
                features=feat_raw,
                labels=all_chain_labels,
                batch_idx=0,
                data_idx=my_seed,
                is_distillation=False
            )
            # print out the number of chains of the protein
            if lab:
                print("chain number", len(lab))
            featd, _ = utils.prepare_features(feat, Job)
            my_seed = random.randint(0, 1000000)  # Generate a random seed
            # my_seed = 10
            # diffuse inputs to noisy structure
            if featd["diffusion_t"] == 1.0:
                if "motif" in Job and Job["motif"] == True:
                    logging.info(f"add noise {featd['diffusion_t']} to backbone frame...")
                    featd = diffuse_inputs(featd, diffuser, my_seed, config.diffusion, task="predict")
                else:
                    logging.info(f"set backbone frame to be prior...")
                    featd = utils.set_prior(featd, diffuser, my_seed, config.diffusion)
            else:
                logging.info(f"add noise {featd['diffusion_t']} to backbone frame...")
                featd = diffuse_inputs(featd, diffuser, my_seed, config.diffusion, task="predict")

            batch = utils.prepare_batch(featd, lab, args)
            process_replica(args, model, batch, gpu_diffuser, config, Job, output_traj_dir, job_name, replica, res1_indices, res2_indices, cutoff_list, eta = 0.8, g_decay = 1.0, decay_radius = 5, decay_lambda = 0.1)
        print("Denoising inference completed!")
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path arguments
    parser.add_argument(
        "-t", "--tasks", type=str, default=None, required=True,
        help=r"""
    path to a json file specifying the task to conduct. The keyword arguments are illustrated in ufconf/README.md.
        """
    )
    parser.add_argument(
        "-i", "--input_pdbs", type=str, default=None,
        help="directory of input reference pdbs."
    )
    parser.add_argument(
        "-msa", "--msa_file_path", type=str, default=None,
        help="the path for existing MSA file"
    )
    parser.add_argument(
        "-o", "--outputs", type=str, default=None, required=True,
        help="directory of outputs."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None,
        help="path to model checkpoint."
    )
    parser.add_argument(
        "-st", "--start", type=int, default=0,
        help="replica number start from"
    )

    # model arguments
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="the device to run the model. Please at any chance use GPUs. (default: `cuda:0`)."
    )
    parser.add_argument(
        "--chunk-size", type=str, default="1024",
        help="chunk size of the model"
    )
    parser.add_argument(
        "--model", type=str, default="ufconf_af2_v3",
        help="the name of the model. (default: `ufconf_af2_v3`)."
    )
    parser.add_argument(
        "--data_idx", type=str, default="0",
        help="specify random seed for MSA."
    )
    parser.add_argument(
        "--use_exist_msa", action="store_true",
        help="if set, then use existing MSA (default: not set)."
    )
    parser.add_argument(
        "--use_guidance", action="store_true",
        help="if set, then use existing guidance (default: not set)."
    )
    parser.add_argument(
        "--from_fasta", action=argparse.BooleanOptionalAction, default=True,
        help="if set, then running inference with the input seq (default: set)."
    )
    parser.add_argument(
        "--from_pdb",  action="store_true",
        help="if set, then running inference with MSA generated on the fly from pdb file (default: set)."
    )
    parser.add_argument(
        "--from_cif", action="store_true",
        help="if set, then running inference with MSA generated on the fly from cif file(default: not set)."
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="if set, then bfloat16 is used. (default: not set)."
    )
    args = parser.parse_args()
    main(args)
