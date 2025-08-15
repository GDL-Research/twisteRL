# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import importlib
import json
import torch
from huggingface_hub import HfApi, snapshot_download
from loguru import logger

def dynamic_import(path):
    try:
        module_name, attr_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{path}'. Error: {e}")

def json_load_tuples(dct):
    if "__tuple_list__" in dct:
        return [tuple(item) for item in dct["list"]]
    return dct

import fnmatch

def validate_algorithm_from_hub(repo_id: str, revision: str = "main"):
    # List of required file patterns to validate model
    REQUIRED_FILES = {
        "*.json",
        "*.pt"
    }
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, revision=revision)
    except:
        return {
            "is_valid": False,
            "missing": ["<repo not found>"]
        }
    files_set = set(files)
    # Check for required file patterns
    missing = []
    for pattern in REQUIRED_FILES:
        if not any(fnmatch.fnmatch(file, pattern) for file in files_set):
            missing.append(pattern)

    is_valid = len(missing) == 0
    return {
        "is_valid": is_valid,
        "missing": missing
    }

def pull_hub_algorithm(repo_id, model_path = "../models/", revision = "main", validate = False):
    validate_algorithm = validate_algorithm_from_hub(repo_id)
    if validate and not validate_algorithm["is_valid"]:
        logger.info(validate_algorithm)
        return False
    try:
        local_repo_path = snapshot_download(
                repo_id,
                cache_dir = model_path,
                allow_patterns = ["*.json", "*.pt"],
                revision = revision,
                force_download=False
            )
        logger.info(f"Model files are now in: {local_repo_path}")
        return local_repo_path
    except: 
        return False

def prepare_algorithm(config, run_path=None, load_checkpoint_path=None):
    # Import env class and make env
    env_cls = dynamic_import(config["env_cls"])
    env = env_cls(**config["env"])

    # Import policy class and make policy
    policy_cls = dynamic_import(config["policy_cls"])
    obs_perms, act_perms = env.twists()
    policy = policy_cls(
        env.obs_shape(),
        env.num_actions(),
        **config["policy"],
        obs_perms=obs_perms,
        act_perms=act_perms,
    )
    if load_checkpoint_path is not None:
        policy.load_state_dict(
            torch.load(
                open(load_checkpoint_path, "rb"), map_location=torch.device("cpu")
            )
        )

    # Import algo class and make algorithm
    algo_cls = dynamic_import(config["algorithm_cls"])
    return algo_cls(env, policy, config["algorithm"], run_path)


def load_config(config_path):
    return json.load(open(config_path), object_hook=json_load_tuples)
