DIAGNOSTIC_SOURCE = '"""Pre-reset diagnostics derived from the successful Colab smoke test."""\nimport json\nfrom pathlib import Path\n\nimport isaacgym\nfrom isaacgym import gymapi, gymutil\nimport torch\nfrom torch import nn\nfrom easydict import EasyDict\n\nfrom legged_gym.envs import LeggedRobot\nfrom legged_gym.utils.helpers import set_seed\n\n\nimport csv\nimport argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--checkpoint\', choices=[\'model_50.pt\'], required=True)\nparser.add_argument(\'--mass-scale\', type=float, choices=[1.0], default=1.0)\nparser.add_argument(\'--output\', type=Path, required=True)\nparser.add_argument(\'--max-steps\', type=int, default=300)\nparser.add_argument(\'--motion-file\', type=Path, required=True)\nparser.add_argument(\'--motion-key\', required=True)\nparser.add_argument(\'--motion-sha256\', required=True)\nparser.add_argument(\'--run-dir\', type=Path, default=Path(\'/content/hr17_training/H2O_seed1_20261116_005911\'))\nparser.set_defaults(controller=\'checkpoint\')\ncli_args = parser.parse_args()\nimport hashlib\nimport numpy as np\n\n\ndef normalized_motion_keys(value):\n    # A one-element tensor index can produce a NumPy string scalar rather than\n    # a one-element array. Flattening preserves a scalar key as one whole string.\n    result = []\n    for key in np.asarray(value).reshape(-1).tolist():\n        if isinstance(key, bytes):\n            key = key.decode(\'utf-8\')\n        assert isinstance(key, str), \'Unexpected motion key type: \' + repr(type(key))\n        result.append(key)\n    return result\n\nfrom collections import Counter\nfrom phc.utils import torch_utils as phc_torch_utils\n\n\nclass DiagnosticRobot(LeggedRobot):\n    """Observe before reset_idx; use the repository\'s original termination logic."""\n\n    def __init__(self, *args, **kwargs):\n        self.diag_enabled = False\n        self.diag_rows = []\n        self.diag_reward_rows = []\n        self.diag_biomechanics = []\n        self.diag_pd = []\n        self.diag_events = {key: [] for key in [\'feet_max_height_for_this_air\', \'in_the_air\', \'dof_vel_limits\']}\n        self.diag_pred = []\n        self.diag_ref = []\n        self.diag_episode = 0\n        super().__init__(*args, **kwargs)\n        self.diag_enabled = True\n        def observe(name, original):\n            def wrapped():\n                event = dict(episode_step=int(self.episode_length_buf[0].item()))\n                def copy(value): return value.detach().cpu().numpy().copy()\n                if name in [\'feet_max_height_for_this_air\', \'in_the_air\']:\n                    event.update(force_z=copy(self.contact_forces[0, self.feet_indices, 2]),\n                                 last_contacts=copy(self.last_contacts[0]),\n                                 foot_z=copy(self._rigid_body_pos[0, self.feet_indices, 2]))\n                if name == \'feet_max_height_for_this_air\':\n                    event.update(last_contacts_filt=copy(self.last_contacts_filt[0]),\n                                 air_max_before=copy(self.feet_air_max_height[0]))\n                if name == \'dof_vel_limits\':\n                    event[\'dof_vel\'] = copy(self.dof_vel[0])\n                value = original()  # exactly one call; preserve its original state updates\n                event[\'raw_value\'] = float(value[0].item())\n                if name == \'feet_max_height_for_this_air\':\n                    event[\'air_max_after\'] = copy(self.feet_air_max_height[0])\n                self.diag_events[name].append(event)\n                return value\n            return wrapped\n        for i, name in enumerate(self.reward_names):\n            if name in self.diag_events:\n                self.reward_functions[i] = observe(name, self.reward_functions[i])\n\n    def _process_rigid_body_props(self, props, env_id):\n        props = super()._process_rigid_body_props(props, env_id)\n        assert env_id == 0\n        self.mass_before_scaling = [float(p.mass) for p in props]\n        for prop in props:\n            prop.mass *= cli_args.mass_scale\n        self.mass_requested = [float(p.mass) for p in props]\n        return props\n\n    def compute_reward(self):\n        if not self.diag_enabled:\n            return super().compute_reward()\n        assert not self.cfg.rewards.only_positive_rewards\n        before = {key: value.detach().clone() for key, value in self.episode_sums.items()}\n        # Call the original implementation exactly once. Read counters before reset_idx.\n        result = super().compute_reward()\n        contributions = {key: (self.episode_sums[key].double() - before[key].double())\n                         for key in before}\n        reconstructed = torch.stack(list(contributions.values())).sum(dim=0)\n        residual = float((reconstructed - self.rew_buf.double()).abs().max().item())\n        assert torch.allclose(reconstructed, self.rew_buf.double(), rtol=1e-4, atol=1e-3), residual\n        assert len(self.diag_rows) == len(self.diag_reward_rows) + 1\n        row = dict(episode_step=int(self.episode_length_buf[0].item()),\n                   reward_total=float(self.rew_buf[0].item()),\n                   reconstruction_residual=residual)\n        row.update({key: float(value[0].item()) for key, value in contributions.items()})\n        self.diag_reward_rows.append(row)\n        return result\n\n    def _compute_torques(self, actions):\n        if not self.diag_enabled:\n            return super()._compute_torques(actions)\n        assert self.cfg.control.control_type == \'P\'\n        assert not self.cfg.domain_rand.randomize_torque_rfi\n        assert not self.cfg.domain_rand.randomize_ctrl_delay\n        assert not self.cfg.control.action_filt\n        def copy(value): return value.detach().cpu().numpy().copy()\n        step = len(self.diag_rows) + 1\n        substep = (self.diag_pd[-1][\'substep\'] + 1\n                   if self.diag_pd and self.diag_pd[-1][\'control_step\'] == step else 1)\n        target = actions * self.cfg.control.action_scale + self.default_dof_pos\n        proportional = self._kp_scale * self.p_gains * (target - self.dof_pos)\n        damping = -self._kd_scale * self.d_gains * self.dof_vel\n        raw_torque = proportional + damping\n        # Exactly one original control call; audit calculations are not returned to the simulator.\n        returned = super()._compute_torques(actions)\n        assert torch.allclose(torch.clip(raw_torque, -self.torque_limits, self.torque_limits),\n                              returned, rtol=1e-5, atol=1e-5)\n        self.diag_pd.append(dict(control_step=step, substep=substep,\n            policy_raw_action=copy(self.diag_pd_policy_action[0]),\n            torque_input_action=copy(actions[0]), target_q=copy(target[0]),\n            q=copy(self.dof_pos[0]), qdot=copy(self.dof_vel[0]),\n            kp_scale=copy(self._kp_scale[0]), kd_scale=copy(self._kd_scale[0]),\n            proportional_torque=copy(proportional[0]), damping_torque=copy(damping[0]),\n            unclipped_torque=copy(raw_torque[0]), returned_torque=copy(returned[0])))\n        return returned\n\n    def check_termination(self):\n        super().check_termination()\n        if not self.diag_enabled:\n            return\n        assert self.num_envs == 1\n\n        offset = self.env_origins + self.env_origins_init_3Doffset\n        motion_time = self.episode_length_buf * self.dt + self.motion_start_times\n        reference = self._get_state_from_motionlib_cache_trimesh(\n            self.motion_ids, motion_time, offset=offset\n        )\n        positions = self._rigid_body_pos\n        rotations = self._rigid_body_rot\n        extended = phc_torch_utils.my_quat_rotate(\n            rotations[:, self.extend_body_parent_ids].reshape(-1, 4),\n            self.extend_body_pos.reshape(-1, 3),\n        ).reshape(1, -1, 3) + positions[:, self.extend_body_parent_ids]\n        predicted = torch.cat((positions, extended), dim=1)\n        target = reference[\'rg_pos_t\']\n        def copy(value): return value.detach().cpu().numpy().copy()\n        assert reference[\'dof_vel\'].shape == self.dof_vel.shape\n        self.diag_biomechanics.append(dict(\n            episode_step=int(self.episode_length_buf[0].item()),\n            robot_foot_pos=copy(positions[0, self.feet_indices]),\n            reference_foot_pos=copy(reference[\'rg_pos\'][0, self.feet_indices]),\n            foot_contact_force=copy(self.contact_forces[0, self.feet_indices]),\n            robot_dof_pos=copy(self.dof_pos[0]), robot_dof_vel=copy(self.dof_vel[0]),\n            reference_dof_pos=copy(reference[\'dof_pos\'][0]), reference_dof_vel=copy(reference[\'dof_vel\'][0]),\n            torques=copy(self.torques[0]),\n        ))\n        assert predicted.shape == target.shape, (predicted.shape, target.shape)\n        assert torch.isfinite(predicted).all() and torch.isfinite(target).all()\n\n        global_error = (predicted - target).norm(dim=-1).mean().item() * 1000\n        local_error = ((predicted - predicted[:, [0]])\n                       - (target - target[:, [0]])).norm(dim=-1).mean().item() * 1000\n        body_distance = (positions - reference[\'rg_pos\']).norm(dim=-1).mean().item()\n\n        contacts = self.contact_forces[0, self.termination_contact_indices].norm(dim=-1)\n        hit_indices = self.termination_contact_indices[contacts > 1.0].tolist()\n        hit_names = [self._body_list[int(i)] for i in hit_indices]\n        reasons = []\n        asset = self.cfg.asset\n        scales = asset.termination_scales\n        if hit_names:\n            reasons.append(\'body_contact\')\n        if asset.terminate_by_gravity:\n            if abs(self.projected_gravity[0, 0].item()) > scales.gravity_x:\n                reasons.append(\'gravity_x\')\n            if abs(self.projected_gravity[0, 1].item()) > scales.gravity_y:\n                reasons.append(\'gravity_y\')\n        if asset.terminate_by_low_height and self.root_states[0, 2].item() < scales.base_height:\n            reasons.append(\'low_height\')\n        if asset.terminate_by_ref_motion_distance and self._recovery_counter[0].item() <= 0:\n            if body_distance > scales.max_ref_motion_distance:\n                reasons.append(\'reference_distance\')\n        if bool(self.time_out_buf[0].item()):\n            reasons.append(\'motion_timeout\')\n        done = bool(self.reset_buf[0].item())\n        if done and not reasons:\n            reasons.append(\'other_termination\')\n\n        self.diag_rows.append(dict(\n            frame=len(self.diag_rows) + 1,\n            episode=self.diag_episode,\n            episode_step=int(self.episode_length_buf[0].item()),\n            motion_time_s=float(motion_time[0].item()),\n            global_mpjpe_mm=global_error,\n            root_relative_mpjpe_mm=local_error,\n            mean_body_distance_m=body_distance,\n            root_height_m=float(self.root_states[0, 2].item()),\n            gravity_x=float(self.projected_gravity[0, 0].item()),\n            gravity_y=float(self.projected_gravity[0, 1].item()),\n            done=done,\n            reasons=\';\'.join(reasons) if done else \'\',\n            contact_bodies=\';\'.join(hit_names) if done else \'\',\n        ))\n        self.diag_pred.append(predicted[0].detach().cpu().numpy().copy())\n        self.diag_ref.append(target[0].detach().cpu().numpy().copy())\n        self.diag_rows[-1].update(\n            requested_action_abs_max=self.diag_action_abs_max,\n            action_clip_fraction=self.diag_action_clip_fraction,\n            joint_target_error_rad=float((self.diag_joint_target - self.dof_pos).abs().mean().item()),\n            torque_abs_max=float(self.torques.abs().max().item()),\n            torque_limit_fraction=float((self.torques.abs() >= 0.999 * self.torque_limits).float().mean().item()),\n        )\n        if done:\n            self.diag_episode += 1\n\n\nrepo = Path("/content/hr17_repo")\nfolder = cli_args.run_dir\nmetadata = json.loads((folder / \'run_metadata.json\').read_text())\nassert metadata[\'status\'] == \'completed\'\nassert metadata[\'seed\'] in [1, 2, 3] and metadata[\'iterations\'] == 200\nassert hashlib.sha256((folder / \'model_200.pt\').read_bytes()).hexdigest() == metadata[\'final_checkpoint_sha256\']\noutput = cli_args.output\noutput.mkdir(parents=True, exist_ok=False)\nimport shutil\nshutil.copy2(__file__, output / Path(__file__).name)\nfor filename in (\'run_metadata.json\', \'env_cfg.json\', \'train_cfg.json\'):\n    shutil.copy2(folder / filename, output / (\'source_\' + filename))\n\n# Đọc lại cấu hình gốc.\ncfg = EasyDict(json.loads((folder / "env_cfg.json").read_text()))\ntrain_cfg = json.loads((folder / "train_cfg.json").read_text())\n\n# Chạy một môi trường, không mở cửa sổ.\ncfg.seed = 1\ncfg.train.seed = 1\ncfg.num_envs = 1\ncfg.env.num_envs = 1\ncfg.headless = True\ncfg.sim_device = "cuda:0"\ncfg.rl_device = "cuda:0"\ncfg.test = True\ncfg.env.test = True\ncfg.env.im_eval = False\ncfg.train.distill = False\ncfg.use_wandb = False\ncfg.viewer.debug_viz = False\ncfg.motion.visualize = False\ncfg.motion.realtime_vr_keypoints = False\ncfg.rewards.penalty_scale = 1.0\n\n# Đường dẫn tới mô hình và motion.\nresources = repo / "legged_gym" / "resources"\ncfg.asset.file = str(resources / "robots/h1/urdf/h1.urdf")\ncfg.motion.motion_file = str(cli_args.motion_file)\nassert hashlib.sha256(cli_args.motion_file.read_bytes()).hexdigest() == cli_args.motion_sha256\ncfg.motion.skeleton_file = str(resources / "robots/h1/xml/h1.xml")\ncfg.motion.marker_file = str(\n    resources / "objects/Marker/traj_marker.urdf"\n)\n\nassert Path(cfg.asset.file).exists(), f"Thiếu file: {cfg.asset.file}"\nfor name in ("motion_file", "skeleton_file"):\n    assert Path(cfg.motion[name]).exists(), (\n        f"Thiếu file: {cfg.motion[name]}"\n    )\n\n# Địa hình phẳng dạng trimesh nhỏ.\ncfg.terrain.curriculum = False\ncfg.terrain.mesh_type = "trimesh"\ncfg.terrain.num_rows = 1\ncfg.terrain.num_cols = 1\ncfg.terrain.max_init_terrain_level = 0\ncfg.terrain.border_size = 5\ncfg.terrain.terrain_types = ["flat"]\ncfg.terrain.terrain_proportions = [1.0]\n\n# Chỉ tắt các cờ ngẫu nhiên kiểu bool.\ncfg.noise.add_noise = False\nfor key, value in list(cfg.domain_rand.items()):\n    if key.startswith("randomize_") and isinstance(value, bool):\n        cfg.domain_rand[key] = False\n\nassert isinstance(\n    cfg.domain_rand.randomize_link_body_names, list\n), "Danh sách khâu robot phải được giữ nguyên"\n\ncfg.domain_rand.push_robots = False\ncfg.domain_rand.motion_package_loss = False\ncfg.domain_rand.born_offset = False\ncfg.domain_rand.born_heading_randomization = False\ncfg.motion.resample_motions_for_envs = False\n\n(output / "config_requested.json").write_text(\n    json.dumps(cfg, indent=2)\n)\n\nset_seed(cfg.seed)\n\nsim_params = gymapi.SimParams()\ngymutil.parse_sim_config(cfg.sim, sim_params)\nsim_params.use_gpu_pipeline = True\nsim_params.physx.use_gpu = True\n\n# Khôi phục actor, không nạp optimizer.\ncheckpoint = torch.load(\n    folder / cli_args.checkpoint,\n    map_location="cpu",\n    weights_only=True,\n)\n\ndims = (\n    [cfg.env.num_observations]\n    + train_cfg["policy"]["actor_hidden_dims"]\n    + [cfg.env.num_actions]\n)\nassert train_cfg["policy"]["activation"].lower() == "elu"\n\nlayers = []\nfor i in range(len(dims) - 1):\n    layers.append(nn.Linear(dims[i], dims[i + 1]))\n    if i < len(dims) - 2:\n        layers.append(nn.ELU())\n\nactor = nn.Sequential(*layers)\nactor.load_state_dict({\n    key[len("actor."):]: value\n    for key, value in checkpoint["model_state_dict"].items()\n    if key.startswith("actor.")\n}, strict=True)\nactor = actor.to("cuda:0").eval()\n\nprint("ĐANG KHỞI TẠO MÔI TRƯỜNG H2O...")\nenv = DiagnosticRobot(\n    cfg=cfg,\n    sim_params=sim_params,\n    physics_engine=gymapi.SIM_PHYSX,\n    sim_device="cuda:0",\n    headless=True,\n)\n\ntry:\n    # Read the actual actor properties back from Isaac Gym before evaluating.\n    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])\n    actual = np.asarray([float(p.mass) for p in props])\n    original = np.asarray(env.mass_before_scaling)\n    requested = np.asarray(env.mass_requested)\n    assert (original > 0).any() and np.isfinite(actual).all()\n    assert np.allclose(requested, original * cli_args.mass_scale, rtol=1e-5, atol=1e-5)\n    assert np.allclose(actual, requested, rtol=1e-4, atol=1e-5), (actual, requested)\n    shapes = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])\n    audit = dict(\n        mass_scale=cli_args.mass_scale,\n        body_names=env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]),\n        original_mass_kg=original.tolist(), requested_mass_kg=requested.tolist(),\n        actual_mass_kg=actual.tolist(), actual_total_mass_kg=float(actual.sum()),\n        original_total_mass_kg=float(original.sum()),\n        mass_readback_verified=True,\n        inertia_recomputed_by_repository=True,\n        actor_shape_friction=[float(p.friction) for p in shapes],\n        terrain_static_friction=float(cfg.terrain.static_friction),\n        terrain_dynamic_friction=float(cfg.terrain.dynamic_friction),\n        control_dt_s=float(env.dt), seed=int(cfg.seed),\n    )\n    (output / \'physics_audit.json\').write_text(json.dumps(audit, indent=2))\n    print(\'PHYSICS VERIFIED:\', cli_args.mass_scale, \'| total mass kg:\', actual.sum())\n    obs = env.get_observations()\n    np.savez_compressed(output / \'initial_state.npz\',\n                        root=env.root_states.detach().cpu().numpy(),\n                        dof=env.dof_pos.detach().cpu().numpy(),\n                        observation=obs.detach().cpu().numpy())\n    np.savez_compressed(output / \'initial_biomechanics.npz\',\n        robot_foot_pos=env._rigid_body_pos[0, env.feet_indices].detach().cpu().numpy(),\n        foot_contact_force=env.contact_forces[0, env.feet_indices].detach().cpu().numpy(),\n        robot_dof_pos=env.dof_pos[0].detach().cpu().numpy(),\n        robot_dof_vel=env.dof_vel[0].detach().cpu().numpy(),\n        motion_start_s=env.motion_start_times.detach().cpu().numpy())\n    diagnostic_offset = (env.env_origins + env.env_origins_init_3Doffset).detach().clone()\n    diagnostic_motion_ids = env.motion_ids.detach().clone()\n    current_keys = normalized_motion_keys(env._motion_lib.curr_motion_keys)\n    file_keys = normalized_motion_keys(env._motion_lib._motion_data_keys)\n    identity_audit = dict(\n        expected_motion_key=cli_args.motion_key,\n        selected_motion_keys=current_keys,\n        file_motion_keys=file_keys,\n        selected_key_container_type=type(env._motion_lib.curr_motion_keys).__name__,\n        selected_key_container_shape=list(np.asarray(env._motion_lib.curr_motion_keys).shape),\n        unique_motion_count=int(env._motion_lib._num_unique_motions),\n    )\n    (output / \'motion_identity_audit.json\').write_text(json.dumps(identity_audit, indent=2))\n    assert current_keys == [cli_args.motion_key], identity_audit\n    assert file_keys == [cli_args.motion_key], identity_audit\n    assert env._motion_lib._num_unique_motions == 1, identity_audit\n    diagnostic_motion_duration = float(env.motion_len[0].item())\n    assert abs(float(env.motion_start_times[0].item())) < 1e-7\n    assert 0 < diagnostic_motion_duration <= cli_args.max_steps * float(env.dt)\n    actor_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}\n    assert obs.shape == (1, 138)\n    print(\'CHECKPOINT:\', cli_args.checkpoint, \'| first episode only; up to\', cli_args.max_steps, \'steps\')\n    with torch.inference_mode():\n        for step in range(cli_args.max_steps):\n            assert torch.isfinite(obs).all(), \'Observation contains NaN/Inf\'\n            if cli_args.controller == \'checkpoint\':\n                actions = actor(obs)\n            elif cli_args.controller == \'zero\':\n                # Zero action means PD holds default joint targets, not zero torque.\n                actions = torch.zeros_like(env.dof_pos)\n            else:\n                assert cfg.control.control_type == \'P\'\n                assert cfg.control.action_scale > 0\n                offset = env.env_origins + env.env_origins_init_3Doffset\n                next_time = (env.episode_length_buf + 1) * env.dt + env.motion_start_times\n                reference = env._get_state_from_motionlib_cache_trimesh(\n                    env.motion_ids, next_time, offset=offset\n                )\n                actions = (reference[\'dof_pos\'] - env.default_dof_pos) / cfg.control.action_scale\n            env.diag_pd_policy_action = actions.detach().clone()\n            clipped = torch.clamp(actions, -cfg.normalization.clip_actions,\n                                  cfg.normalization.clip_actions)\n            env.diag_joint_target = (clipped * cfg.control.action_scale\n                                     + env.default_dof_pos).detach().clone()\n            env.diag_action_abs_max = float(actions.abs().max().item())\n            env.diag_action_clip_fraction = float((actions != clipped).float().mean().item())\n            assert torch.isfinite(actions).all(), \'Action contains NaN/Inf\'\n            previous_count = len(env.diag_rows)\n            obs, _, rewards, dones, infos = env.step(actions)\n            assert len(env.diag_rows) == previous_count + 1\n            assert torch.isfinite(rewards).all(), \'Reward contains NaN/Inf\'\n            row = env.diag_rows[-1]\n            row[\'reward\'] = float(rewards[0].item())\n            if row[\'done\']:\n                print(f"Reset at frame {row[\'frame\']}, episode step {row[\'episode_step\']}: "\n                      f"{row[\'reasons\']} | contacts={row[\'contact_bodies\']} | "\n                      f"global error={row[\'global_mpjpe_mm\']:.2f} mm")\n                break\n            if row[\'motion_time_s\'] >= diagnostic_motion_duration:\n                print(\'REFERENCE MOTION END REACHED; stopping first episode\')\n                break\n\n    rows = env.diag_rows\n    assert len(env.diag_pd) == len(rows) * int(cfg.control.decimation)\n    np.savez_compressed(output / \'pd_substeps.npz\', **{\n        key: np.asarray([row[key] for row in env.diag_pd]) for key in env.diag_pd[0]})\n    (output / \'pd_metadata.json\').write_text(json.dumps(dict(\n        dof_names=list(env.dof_names), control_type=cfg.control.control_type,\n        action_scale=float(cfg.control.action_scale), clip_actions=float(cfg.normalization.clip_actions),\n        default_q=env.default_dof_pos[0].detach().cpu().tolist(),\n        p_gains=env.p_gains.detach().cpu().tolist(), d_gains=env.d_gains.detach().cpu().tolist(),\n        torque_limits_Nm=env.torque_limits.detach().cpu().tolist(),\n        decimation=int(cfg.control.decimation), control_dt_s=float(env.dt),\n        simulation_dt_s=float(env.sim_params.dt),\n        note=\'Each substep snapshot is before simulation integration. Returned torque is the original controller output. \'\n             \'A normalized action is an offset from default_q, not an absolute angle.\',\n    ), indent=2))\n    assert len(env.diag_biomechanics) == len(rows)\n    np.savez_compressed(output / \'biomechanics_before_reset.npz\', **{\n        key: np.asarray([row[key] for row in env.diag_biomechanics])\n        for key in env.diag_biomechanics[0]})\n    for name, events in env.diag_events.items():\n        assert len(events) == len(rows), (name, len(events), len(rows))\n        np.savez_compressed(output / (\'reward_event_\' + name + \'.npz\'), **{\n            key: np.asarray([row[key] for row in events]) for key in events[0]})\n    (output / \'biomechanics_metadata.json\').write_text(json.dumps(dict(\n        foot_body_indices=env.feet_indices.detach().cpu().tolist(),\n        foot_body_names=[env._body_list[int(i)] for i in env.feet_indices],\n        dof_names=list(env.dof_names),\n        velocity_limits_rad_s=env.dof_vel_limits.detach().cpu().tolist(),\n        soft_velocity_limit_factor=float(cfg.rewards.soft_dof_vel_limit),\n        desired_foot_max_height_m=float(cfg.rewards.desired_feet_max_height_for_this_air),\n        contact_force_threshold_N=1.0,\n        position_scope=\'world z of rigid-body origins; not sole clearance above terrain\',\n        reference_contact_force_available=False,\n        timing=\'snapshots before reset; reward-event snapshots taken at the original function call\',\n    ), indent=2))\n    # Query the complete reference only AFTER all rollout samples have been copied.\n    motion_id = int(diagnostic_motion_ids[0].item())\n    count = int(env._motion_lib._motion_num_frames[motion_id].item())\n    frame_dt = float(env._motion_lib._motion_dt[motion_id].item())\n    profile = []\n    for index in range(count):\n        time = torch.full_like(env.motion_start_times, index * frame_dt)\n        state = env._get_state_from_motionlib_cache_trimesh(\n            diagnostic_motion_ids, time, offset=diagnostic_offset)\n        profile.append(dict(time_s=index * frame_dt,\n            foot_pos=state[\'rg_pos\'][0, env.feet_indices].detach().cpu().numpy().copy(),\n            dof_pos=state[\'dof_pos\'][0].detach().cpu().numpy().copy(),\n            dof_vel=state[\'dof_vel\'][0].detach().cpu().numpy().copy()))\n    np.savez_compressed(output / \'reference_full_motion.npz\', **{\n        key: np.asarray([sample[key] for sample in profile]) for key in profile[0]})\n    assert len(env.diag_reward_rows) == len(rows)\n    with (output / \'reward_terms.csv\').open(\'w\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(env.diag_reward_rows[0]))\n        writer.writeheader()\n        writer.writerows(env.diag_reward_rows)\n    (output / \'reward_scale_audit.json\').write_text(json.dumps(dict(\n        runtime_scales={key: float(value) for key, value in env.reward_scales.items()},\n        penalty_scale=float(cfg.rewards.penalty_scale),\n        penalty_reward_names=list(cfg.rewards.penalty_reward_names),\n        only_positive_rewards=bool(cfg.rewards.only_positive_rewards),\n        contribution_method=\'episode_sums delta around one original compute_reward call, before reset; float32 counter quantization possible\',\n        total_reconstruction_atol=1e-3, total_reconstruction_rtol=1e-4,\n    ), indent=2))\n    assert 1 <= len(rows) <= cli_args.max_steps\n    assert all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())\n    with (output / \'frames.csv\').open(\'w\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))\n        writer.writeheader()\n        writer.writerows(rows)\n    np.savez_compressed(output / \'positions_before_reset.npz\',\n                        robot=np.stack(env.diag_pred), reference=np.stack(env.diag_ref))\n    reason_counts = Counter(reason for row in rows if row[\'done\']\n                            for reason in row[\'reasons\'].split(\';\') if reason)\n    summary = dict(\n        purpose=\'fixed model_50, intact single motion, first episode until termination or motion end; not paper evaluation\',\n        training_seed=int(metadata[\'seed\']),\n        evaluation_seed=1,\n        intervention_mode=\'none\',\n        motion_key=cli_args.motion_key,\n        motion_file_sha256=cli_args.motion_sha256,\n        mass_scale=cli_args.mass_scale,\n        actor_weights_unchanged=True,\n        maximum_control_steps=cli_args.max_steps,\n        controller=cli_args.controller,\n        checkpoint_name=cli_args.checkpoint,\n        checkpoint_used_for_actions=(cli_args.controller == \'checkpoint\'),\n        max_requested_action=float(max(row[\'requested_action_abs_max\'] for row in rows)),\n        mean_action_clip_fraction=float(np.mean([row[\'action_clip_fraction\'] for row in rows])),\n        mean_torque_limit_fraction=float(np.mean([row[\'torque_limit_fraction\'] for row in rows])),\n        metric_scope=\'first episode only, including terminal frame; compare MPJPE on common nonterminal prefix\',\n        checkpoint_provenance=\'user fresh PPO run; archived hyperparameter source provenance unknown\',\n        training_run=folder.name,\n        training_initialization=metadata[\'initialization\'],\n        deterministic_actor=True,\n        checkpoint_sha256=hashlib.sha256((folder / cli_args.checkpoint).read_bytes()).hexdigest(),\n        seed=cfg.seed,\n        control_dt_s=float(env.dt),\n        frame_count=len(rows),\n        reset_count=sum(row[\'done\'] for row in rows),\n        first_reset_step=next((r[\'episode_step\'] for r in rows if r[\'done\']), None),\n        first_reset_time_s=next((r[\'motion_time_s\'] for r in rows if r[\'done\']), None),\n        reset_reason_counts=dict(reason_counts),\n        observed_frame_global_mpjpe_mm=float(np.mean([r[\'global_mpjpe_mm\'] for r in rows])),\n        observed_frame_root_relative_mpjpe_mm=float(np.mean([r[\'root_relative_mpjpe_mm\'] for r in rows])),\n        num_body_points=int(env.diag_pred[0].shape[0]),\n        reference_distance_termination_threshold_m=float(cfg.asset.termination_scales.max_ref_motion_distance),\n        success_rate=None,\n        note=\'Multiple causes may occur in one reset; success rate is not measured here.\',\n    )\n    last = rows[-1]\n    failure = bool(last[\'done\']) and any(reason != \'motion_timeout\' for reason in last[\'reasons\'].split(\';\') if reason)\n    reached = last[\'motion_time_s\'] >= diagnostic_motion_duration\n    summary.update(motion_duration_s=diagnostic_motion_duration, intervention_control_steps=0,\n        failure_time_s=last[\'motion_time_s\'] if failure else None,\n        reached_motion_end_without_recorded_failure=bool(reached and not failure),\n        stop_reason=\'termination\' if failure else (\'motion_end\' if reached else \'step_limit\'))\n    (output / \'summary.json\').write_text(json.dumps(summary, indent=2))\n    print(\'\\nDIAGNOSTIC SUMMARY:\')\n    print(json.dumps(summary, indent=2))\n    print(\'CHAN DOAN RESET HOAN TAT\')\n    print(\'Results:\', output)\nfinally:\n    env.gym.destroy_sim(env.sim)\n'

"""Frozen six-motion evaluation. Default mode prepares data without GPU imports."""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def capture(command, **kwargs):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=90, **kwargs)
    assert result.returncode == 0, result.stdout
    return result.stdout


def prepare(data):
    manifest = load(data / 'manifest.json')
    assert manifest['status'] == 'data_prepared_not_evaluated'
    rows = manifest['motions']
    assert len(rows) == 6 and {r['id'] for r in rows} == {'id01', 'id02', 'id03', 'ood01', 'ood02', 'ood03'}
    for row in rows:
        assert row['file'] == row['id'] + '.pkl'
        assert sha(data / row['file']) == row['sha256'], row['file']
        assert row['duration_s'] > 0 and row['fps'] == 30
        if row['group'] == 'held_out_motion':
            assert row['motion_key'] not in manifest['training_pool_keys']
    ast.parse(DIAGNOSTIC_SOURCE)
    print('DU LIEU VA SCRIPT DA SAN SANG; CHUA CHAY MO PHONG.', flush=True)
    for row in rows:
        # One extra step accommodates float32 duration / control-dt roundoff.
        print(row['id'], '|', row['duration_s'], 's | at most',
              math.ceil(row['duration_s'] / .02) + 1, 'steps', flush=True)
    return manifest


def summarize(output, motions):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, motion in zip(axes.flat, motions):
        folder = output / motion['id']
        summary = load(folder / 'summary.json')
        assert summary['checkpoint_sha256'] == CHECKPOINT_SHA
        with (folder / 'frames.csv').open(newline='') as handle:
            frames = list(csv.DictReader(handle))
        positions = np.load(folder / 'positions_before_reset.npz')
        robot = positions['robot'].astype(np.float64)
        reference = positions['reference'].astype(np.float64)
        assert robot.shape == reference.shape and robot.shape[1:] == (22, 3)
        assert len(robot) == len(frames) > 0
        assert np.isfinite(robot).all() and np.isfinite(reference).all()
        global_errors = np.linalg.norm(robot - reference, axis=-1).mean(axis=-1) * 1000
        local_errors = np.linalg.norm((robot - robot[:, [0]]) - (reference - reference[:, [0]]), axis=-1).mean(axis=-1) * 1000
        np.testing.assert_allclose(global_errors, [float(f['global_mpjpe_mm']) for f in frames], rtol=1e-5, atol=.001)
        np.testing.assert_allclose(local_errors, [float(f['root_relative_mpjpe_mm']) for f in frames], rtol=1e-5, atol=.001)
        terminal = np.array([f['done'] == 'True' for f in frames])
        assert terminal[:-1].sum() == 0
        times = np.array([float(f['motion_time_s']) for f in frames])
        assert np.all(np.diff(times) > 0)
        assert abs(summary['motion_duration_s'] - motion['duration_s']) < 1e-4
        initial = np.load(folder / 'initial_biomechanics.npz')
        assert abs(float(initial['motion_start_s'][0])) < 1e-7
        row = dict(id=motion['id'], group=motion['group'], motion_key=motion['motion_key'],
                   duration_s=motion['duration_s'], observed_steps=len(frames), observed_time_s=float(times[-1]),
                   observed_duration_fraction=min(1., float(times[-1]) / motion['duration_s']),
                   failure_time_s=summary['failure_time_s'], stop_reason=summary['stop_reason'],
                   reset_reasons=frames[-1]['reasons'],
                   reached_motion_end_without_recorded_failure=summary['reached_motion_end_without_recorded_failure'],
                   observed_frame_global_mean_mm=float(global_errors.mean()),
                   observed_frame_global_max_mm=float(global_errors.max()),
                   observed_frame_root_relative_mean_mm=float(local_errors.mean()),
                   observed_frame_root_relative_max_mm=float(local_errors.max()),
                   metric_scope='all observed frames including terminal frame; different clip-specific horizons')
        valid5 = len(frames) >= 5 and not terminal[:5].any()
        row['first5_nonterminal_global_mean_mm'] = float(global_errors[:5].mean()) if valid5 else None
        row['first5_nonterminal_root_relative_mean_mm'] = float(local_errors[:5].mean()) if valid5 else None
        results.append(row)
        ax.plot(times, global_errors, label='Global MPJPE')
        ax.plot(times, local_errors, label='Root-relative MPJPE')
        ax.set(title=motion['id'] + ' | ' + row['stop_reason'], xlabel='Simulated time (s)', ylabel='Error (mm)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle('Fixed model_50 | 3 training-pool + 3 held-out clips | first episode')
    fig.savefig(output / 'motion_errors.png', dpi=160)
    plt.close(fig)
    with (output / 'motion_results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write(output / 'motion_results.json', dict(results=results, no_training=True,
          paper_success_rate=None, scope='six preselected clips; exploratory evaluation, not paper protocol',
          warning='Observed-frame mean/max values have different horizons. Do not rank ID/OOD groups using these averages alone. First-five-step metrics are available only when all five frames are nonterminal.'))


CHECKPOINT_SHA = '59c0bc4d25d8a3d5b468016ad1081fdfc2c44f41899ad4b530bc9573513d28fc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('/content/hr17_motion_id_ood'))
    parser.add_argument('--run', action='store_true', help='Run physics; requires restored GPU environment')
    args = parser.parse_args()
    manifest = prepare(args.data_dir)
    if not args.run:
        return
    root = Path('/content/hr17_motion_evaluation')
    output = root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='preparing', no_training=True, checkpoint_sha256=CHECKPOINT_SHA,
                  intervention='none', evaluation_seed=1, cases=[m['id'] for m in manifest['motions']])
    try:
        python = Path('/content/hr17_env/bin/python')
        repo = Path('/content/hr17_repo')
        run = Path('/content/hr17_training/H2O_seed1_20261116_005911')
        assert python.is_file() and repo.is_dir(), 'Restore the Python 3.8 / H2O GPU environment first'
        checkpoint = run / 'model_50.pt'
        assert sha(checkpoint) == CHECKPOINT_SHA
        metadata = load(run / 'run_metadata.json')
        assert metadata['status'] == 'completed' and metadata['seed'] == 1
        assert metadata['motion_sha256'] == manifest['training_pool_sha256']
        robot_source = repo / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        source_hash = sha(robot_source)
        assert source_hash == '484b1e3f96f71d77b1c74a17bea195b2773e7010a616db37a65f4cfc12e9ba26', 'Restore the verified gravity fix; do not edit simulator logic'
        assert capture(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).strip() == metadata['repo_commit']
        diff = capture(['git', 'diff', '--no-ext-diff'], cwd=str(repo))
        assert diff.splitlines() == (run / 'repo_changes.patch').read_text().splitlines(), 'Repository differs from training'
        runtime = load(Path('/content/hr17_runtime_environment.json'))
        assert runtime.get('native_h2o_import_verified'), 'Complete the native import verification in the current session'
        env = os.environ.copy()
        env.update(PATH='/content/hr17_env/bin:' + env.get('PATH', ''),
                   LD_LIBRARY_PATH=runtime['LD_LIBRARY_PATH'], MAX_JOBS='2', MPLBACKEND='Agg',
                   PYTHONHASHSEED='1', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
        freeze = capture([str(python), '-m', 'pip', 'freeze'], env=env)
        lines = lambda text: {line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')}
        assert lines(freeze) == lines((run / 'environment_freeze.txt').read_text()), 'Packages differ from saved training environment'
        (output / 'environment_freeze.txt').write_text(freeze)
        (output / 'repo_diff.patch').write_text(diff)
        (output / 'gpu_info.txt').write_text(capture(['nvidia-smi']))
        write(output / 'runtime_environment.json', runtime)
        write(output / 'motion_manifest.json', manifest)
        shutil.copy2(__file__, output / Path(__file__).name)
        child = output / 'evaluate_motion_case.py'
        child.write_text(DIAGNOSTIC_SOURCE, encoding='utf-8')
        inputs = output / 'motion_inputs'
        inputs.mkdir()
        for motion in manifest['motions']:
            shutil.copy2(args.data_dir / motion['file'], inputs / motion['file'])
        status['status'] = 'evaluating'
        for motion in manifest['motions']:
            steps = math.ceil(motion['duration_s'] / .02) + 1
            command = [str(python), '-u', str(child), '--checkpoint', 'model_50.pt',
                       '--run-dir', str(run), '--output', str(output / motion['id']),
                       '--max-steps', str(steps), '--motion-file', str(inputs / motion['file']),
                       '--motion-key', motion['motion_key'], '--motion-sha256', motion['sha256']]
            print('\nEVALUATING', motion['id'], '| maximum', steps, 'steps', flush=True)
            try:
                result = subprocess.run(command, cwd=str(repo), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, timeout=180)
                code, log = result.returncode, result.stdout
            except subprocess.TimeoutExpired as error:
                log = error.stdout or b''
                if isinstance(log, bytes):
                    log = log.decode(errors='replace')
                code, log = 124, log + '\nWALL-CLOCK TIMEOUT (180 s), not a robot failure metric\n'
            (output / (motion['id'] + '.log')).write_text(log, encoding='utf-8')
            print(log[-2500:], flush=True)
            assert code == 0, motion['id'] + ' failed; inspect the saved log'
        summarize(output, manifest['motions'])
        assert sha(checkpoint) == CHECKPOINT_SHA
        assert sha(robot_source) == source_hash
        assert capture(['git', 'diff', '--no-ext-diff'], cwd=str(repo)).splitlines() == diff.splitlines()
        status.update(status='completed', checkpoint_unchanged=True, repository_unchanged=True)
        print('SIX-MOTION EVALUATION COMPLETED', flush=True)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / 'experiment_status.json', status)
        write(output / 'SHA256_manifest.json', {p.relative_to(output).as_posix(): sha(p)
              for p in output.rglob('*') if p.is_file()})
        archive = shutil.make_archive(str(output), 'zip', root_dir=str(output))
        with zipfile.ZipFile(archive) as package:
            assert package.testzip() is None
        (root / 'latest_archive.txt').write_text(archive)
        print('RESULT ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
