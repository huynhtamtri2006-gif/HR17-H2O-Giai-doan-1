"""Paired first-step elbow intervention followed by the original actor for the first episode; no training."""
DIAGNOSTIC_SOURCE = '"""Pre-reset diagnostics derived from the successful Colab smoke test."""\nimport json\nfrom pathlib import Path\n\nimport isaacgym\nfrom isaacgym import gymapi, gymutil\nimport torch\nfrom torch import nn\nfrom easydict import EasyDict\n\nfrom legged_gym.envs import LeggedRobot\nfrom legged_gym.utils.helpers import set_seed\n\n\nimport csv\nimport argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--checkpoint\', choices=[\'model_initial.pt\', \'model_200.pt\'], required=True)\nparser.add_argument(\'--mass-scale\', type=float, choices=[1.0], default=1.0)\nparser.add_argument(\'--output\', type=Path, required=True)\nparser.add_argument(\'--max-steps\', type=int, choices=[300], default=300)\nparser.add_argument(\'--elbow-mode\', choices=[\'original\', \'hold_initial\'], required=True)\nparser.add_argument(\'--run-dir\', type=Path, default=Path(\'/content/hr17_training/H2O_seed1_20261116_005911\'))\nparser.set_defaults(controller=\'checkpoint\')\ncli_args = parser.parse_args()\nimport hashlib\nimport numpy as np\nfrom collections import Counter\nfrom phc.utils import torch_utils as phc_torch_utils\n\n\nclass DiagnosticRobot(LeggedRobot):\n    """Observe before reset_idx; use the repository\'s original termination logic."""\n\n    def __init__(self, *args, **kwargs):\n        self.diag_enabled = False\n        self.diag_rows = []\n        self.diag_reward_rows = []\n        self.diag_biomechanics = []\n        self.diag_pd = []\n        self.diag_events = {key: [] for key in [\'feet_max_height_for_this_air\', \'in_the_air\', \'dof_vel_limits\']}\n        self.diag_pred = []\n        self.diag_ref = []\n        self.diag_episode = 0\n        super().__init__(*args, **kwargs)\n        self.diag_enabled = True\n        def observe(name, original):\n            def wrapped():\n                event = dict(episode_step=int(self.episode_length_buf[0].item()))\n                def copy(value): return value.detach().cpu().numpy().copy()\n                if name in [\'feet_max_height_for_this_air\', \'in_the_air\']:\n                    event.update(force_z=copy(self.contact_forces[0, self.feet_indices, 2]),\n                                 last_contacts=copy(self.last_contacts[0]),\n                                 foot_z=copy(self._rigid_body_pos[0, self.feet_indices, 2]))\n                if name == \'feet_max_height_for_this_air\':\n                    event.update(last_contacts_filt=copy(self.last_contacts_filt[0]),\n                                 air_max_before=copy(self.feet_air_max_height[0]))\n                if name == \'dof_vel_limits\':\n                    event[\'dof_vel\'] = copy(self.dof_vel[0])\n                value = original()  # exactly one call; preserve its original state updates\n                event[\'raw_value\'] = float(value[0].item())\n                if name == \'feet_max_height_for_this_air\':\n                    event[\'air_max_after\'] = copy(self.feet_air_max_height[0])\n                self.diag_events[name].append(event)\n                return value\n            return wrapped\n        for i, name in enumerate(self.reward_names):\n            if name in self.diag_events:\n                self.reward_functions[i] = observe(name, self.reward_functions[i])\n\n    def _process_rigid_body_props(self, props, env_id):\n        props = super()._process_rigid_body_props(props, env_id)\n        assert env_id == 0\n        self.mass_before_scaling = [float(p.mass) for p in props]\n        for prop in props:\n            prop.mass *= cli_args.mass_scale\n        self.mass_requested = [float(p.mass) for p in props]\n        return props\n\n    def compute_reward(self):\n        if not self.diag_enabled:\n            return super().compute_reward()\n        assert not self.cfg.rewards.only_positive_rewards\n        before = {key: value.detach().clone() for key, value in self.episode_sums.items()}\n        # Call the original implementation exactly once. Read counters before reset_idx.\n        result = super().compute_reward()\n        contributions = {key: (self.episode_sums[key].double() - before[key].double())\n                         for key in before}\n        reconstructed = torch.stack(list(contributions.values())).sum(dim=0)\n        residual = float((reconstructed - self.rew_buf.double()).abs().max().item())\n        assert torch.allclose(reconstructed, self.rew_buf.double(), rtol=1e-4, atol=1e-3), residual\n        assert len(self.diag_rows) == len(self.diag_reward_rows) + 1\n        row = dict(episode_step=int(self.episode_length_buf[0].item()),\n                   reward_total=float(self.rew_buf[0].item()),\n                   reconstruction_residual=residual)\n        row.update({key: float(value[0].item()) for key, value in contributions.items()})\n        self.diag_reward_rows.append(row)\n        return result\n\n    def _compute_torques(self, actions):\n        if not self.diag_enabled:\n            return super()._compute_torques(actions)\n        assert self.cfg.control.control_type == \'P\'\n        assert not self.cfg.domain_rand.randomize_torque_rfi\n        assert not self.cfg.domain_rand.randomize_ctrl_delay\n        assert not self.cfg.control.action_filt\n        def copy(value): return value.detach().cpu().numpy().copy()\n        step = len(self.diag_rows) + 1\n        substep = (self.diag_pd[-1][\'substep\'] + 1\n                   if self.diag_pd and self.diag_pd[-1][\'control_step\'] == step else 1)\n        target = actions * self.cfg.control.action_scale + self.default_dof_pos\n        proportional = self._kp_scale * self.p_gains * (target - self.dof_pos)\n        damping = -self._kd_scale * self.d_gains * self.dof_vel\n        raw_torque = proportional + damping\n        # Exactly one original control call; audit calculations are not returned to the simulator.\n        returned = super()._compute_torques(actions)\n        assert torch.allclose(torch.clip(raw_torque, -self.torque_limits, self.torque_limits),\n                              returned, rtol=1e-5, atol=1e-5)\n        self.diag_pd.append(dict(control_step=step, substep=substep,\n            policy_raw_action=copy(self.diag_pd_policy_action[0]),\n            torque_input_action=copy(actions[0]), target_q=copy(target[0]),\n            q=copy(self.dof_pos[0]), qdot=copy(self.dof_vel[0]),\n            kp_scale=copy(self._kp_scale[0]), kd_scale=copy(self._kd_scale[0]),\n            proportional_torque=copy(proportional[0]), damping_torque=copy(damping[0]),\n            unclipped_torque=copy(raw_torque[0]), returned_torque=copy(returned[0])))\n        return returned\n\n    def check_termination(self):\n        super().check_termination()\n        if not self.diag_enabled:\n            return\n        assert self.num_envs == 1\n\n        offset = self.env_origins + self.env_origins_init_3Doffset\n        motion_time = self.episode_length_buf * self.dt + self.motion_start_times\n        reference = self._get_state_from_motionlib_cache_trimesh(\n            self.motion_ids, motion_time, offset=offset\n        )\n        positions = self._rigid_body_pos\n        rotations = self._rigid_body_rot\n        extended = phc_torch_utils.my_quat_rotate(\n            rotations[:, self.extend_body_parent_ids].reshape(-1, 4),\n            self.extend_body_pos.reshape(-1, 3),\n        ).reshape(1, -1, 3) + positions[:, self.extend_body_parent_ids]\n        predicted = torch.cat((positions, extended), dim=1)\n        target = reference[\'rg_pos_t\']\n        def copy(value): return value.detach().cpu().numpy().copy()\n        assert reference[\'dof_vel\'].shape == self.dof_vel.shape\n        self.diag_biomechanics.append(dict(\n            episode_step=int(self.episode_length_buf[0].item()),\n            robot_foot_pos=copy(positions[0, self.feet_indices]),\n            reference_foot_pos=copy(reference[\'rg_pos\'][0, self.feet_indices]),\n            foot_contact_force=copy(self.contact_forces[0, self.feet_indices]),\n            robot_dof_pos=copy(self.dof_pos[0]), robot_dof_vel=copy(self.dof_vel[0]),\n            reference_dof_pos=copy(reference[\'dof_pos\'][0]), reference_dof_vel=copy(reference[\'dof_vel\'][0]),\n            torques=copy(self.torques[0]),\n        ))\n        assert predicted.shape == target.shape, (predicted.shape, target.shape)\n        assert torch.isfinite(predicted).all() and torch.isfinite(target).all()\n\n        global_error = (predicted - target).norm(dim=-1).mean().item() * 1000\n        local_error = ((predicted - predicted[:, [0]])\n                       - (target - target[:, [0]])).norm(dim=-1).mean().item() * 1000\n        body_distance = (positions - reference[\'rg_pos\']).norm(dim=-1).mean().item()\n\n        contacts = self.contact_forces[0, self.termination_contact_indices].norm(dim=-1)\n        hit_indices = self.termination_contact_indices[contacts > 1.0].tolist()\n        hit_names = [self._body_list[int(i)] for i in hit_indices]\n        reasons = []\n        asset = self.cfg.asset\n        scales = asset.termination_scales\n        if hit_names:\n            reasons.append(\'body_contact\')\n        if asset.terminate_by_gravity:\n            if abs(self.projected_gravity[0, 0].item()) > scales.gravity_x:\n                reasons.append(\'gravity_x\')\n            if abs(self.projected_gravity[0, 1].item()) > scales.gravity_y:\n                reasons.append(\'gravity_y\')\n        if asset.terminate_by_low_height and self.root_states[0, 2].item() < scales.base_height:\n            reasons.append(\'low_height\')\n        if asset.terminate_by_ref_motion_distance and self._recovery_counter[0].item() <= 0:\n            if body_distance > scales.max_ref_motion_distance:\n                reasons.append(\'reference_distance\')\n        if bool(self.time_out_buf[0].item()):\n            reasons.append(\'motion_timeout\')\n        done = bool(self.reset_buf[0].item())\n        if done and not reasons:\n            reasons.append(\'other_termination\')\n\n        self.diag_rows.append(dict(\n            frame=len(self.diag_rows) + 1,\n            episode=self.diag_episode,\n            episode_step=int(self.episode_length_buf[0].item()),\n            motion_time_s=float(motion_time[0].item()),\n            global_mpjpe_mm=global_error,\n            root_relative_mpjpe_mm=local_error,\n            mean_body_distance_m=body_distance,\n            root_height_m=float(self.root_states[0, 2].item()),\n            gravity_x=float(self.projected_gravity[0, 0].item()),\n            gravity_y=float(self.projected_gravity[0, 1].item()),\n            done=done,\n            reasons=\';\'.join(reasons) if done else \'\',\n            contact_bodies=\';\'.join(hit_names) if done else \'\',\n        ))\n        self.diag_pred.append(predicted[0].detach().cpu().numpy().copy())\n        self.diag_ref.append(target[0].detach().cpu().numpy().copy())\n        self.diag_rows[-1].update(\n            requested_action_abs_max=self.diag_action_abs_max,\n            action_clip_fraction=self.diag_action_clip_fraction,\n            joint_target_error_rad=float((self.diag_joint_target - self.dof_pos).abs().mean().item()),\n            torque_abs_max=float(self.torques.abs().max().item()),\n            torque_limit_fraction=float((self.torques.abs() >= 0.999 * self.torque_limits).float().mean().item()),\n        )\n        if done:\n            self.diag_episode += 1\n\n\nrepo = Path("/content/hr17_repo")\nfolder = cli_args.run_dir\nmetadata = json.loads((folder / \'run_metadata.json\').read_text())\nassert metadata[\'status\'] == \'completed\'\nassert metadata[\'seed\'] in [1, 2, 3] and metadata[\'iterations\'] == 200\nassert hashlib.sha256((folder / \'model_200.pt\').read_bytes()).hexdigest() == metadata[\'final_checkpoint_sha256\']\noutput = cli_args.output\noutput.mkdir(parents=True, exist_ok=False)\nimport shutil\nshutil.copy2(__file__, output / Path(__file__).name)\nfor filename in (\'run_metadata.json\', \'env_cfg.json\', \'train_cfg.json\'):\n    shutil.copy2(folder / filename, output / (\'source_\' + filename))\n\n# Đọc lại cấu hình gốc.\ncfg = EasyDict(json.loads((folder / "env_cfg.json").read_text()))\ntrain_cfg = json.loads((folder / "train_cfg.json").read_text())\n\n# Chạy một môi trường, không mở cửa sổ.\ncfg.seed = 1\ncfg.train.seed = 1\ncfg.num_envs = 1\ncfg.env.num_envs = 1\ncfg.headless = True\ncfg.sim_device = "cuda:0"\ncfg.rl_device = "cuda:0"\ncfg.test = True\ncfg.env.test = True\ncfg.env.im_eval = False\ncfg.train.distill = False\ncfg.use_wandb = False\ncfg.viewer.debug_viz = False\ncfg.motion.visualize = False\ncfg.motion.realtime_vr_keypoints = False\ncfg.rewards.penalty_scale = 1.0\n\n# Đường dẫn tới mô hình và motion.\nresources = repo / "legged_gym" / "resources"\ncfg.asset.file = str(resources / "robots/h1/urdf/h1.urdf")\ncfg.motion.motion_file = str(resources / "motions/h1/stable_punch.pkl")\ncfg.motion.skeleton_file = str(resources / "robots/h1/xml/h1.xml")\ncfg.motion.marker_file = str(\n    resources / "objects/Marker/traj_marker.urdf"\n)\n\nassert Path(cfg.asset.file).exists(), f"Thiếu file: {cfg.asset.file}"\nfor name in ("motion_file", "skeleton_file"):\n    assert Path(cfg.motion[name]).exists(), (\n        f"Thiếu file: {cfg.motion[name]}"\n    )\n\n# Địa hình phẳng dạng trimesh nhỏ.\ncfg.terrain.curriculum = False\ncfg.terrain.mesh_type = "trimesh"\ncfg.terrain.num_rows = 1\ncfg.terrain.num_cols = 1\ncfg.terrain.max_init_terrain_level = 0\ncfg.terrain.border_size = 5\ncfg.terrain.terrain_types = ["flat"]\ncfg.terrain.terrain_proportions = [1.0]\n\n# Chỉ tắt các cờ ngẫu nhiên kiểu bool.\ncfg.noise.add_noise = False\nfor key, value in list(cfg.domain_rand.items()):\n    if key.startswith("randomize_") and isinstance(value, bool):\n        cfg.domain_rand[key] = False\n\nassert isinstance(\n    cfg.domain_rand.randomize_link_body_names, list\n), "Danh sách khâu robot phải được giữ nguyên"\n\ncfg.domain_rand.push_robots = False\ncfg.domain_rand.motion_package_loss = False\ncfg.domain_rand.born_offset = False\ncfg.domain_rand.born_heading_randomization = False\ncfg.motion.resample_motions_for_envs = False\n\n(output / "config_requested.json").write_text(\n    json.dumps(cfg, indent=2)\n)\n\nset_seed(cfg.seed)\n\nsim_params = gymapi.SimParams()\ngymutil.parse_sim_config(cfg.sim, sim_params)\nsim_params.use_gpu_pipeline = True\nsim_params.physx.use_gpu = True\n\n# Khôi phục actor, không nạp optimizer.\ncheckpoint = torch.load(\n    folder / cli_args.checkpoint,\n    map_location="cpu",\n    weights_only=True,\n)\n\ndims = (\n    [cfg.env.num_observations]\n    + train_cfg["policy"]["actor_hidden_dims"]\n    + [cfg.env.num_actions]\n)\nassert train_cfg["policy"]["activation"].lower() == "elu"\n\nlayers = []\nfor i in range(len(dims) - 1):\n    layers.append(nn.Linear(dims[i], dims[i + 1]))\n    if i < len(dims) - 2:\n        layers.append(nn.ELU())\n\nactor = nn.Sequential(*layers)\nactor.load_state_dict({\n    key[len("actor."):]: value\n    for key, value in checkpoint["model_state_dict"].items()\n    if key.startswith("actor.")\n}, strict=True)\nactor = actor.to("cuda:0").eval()\n\nprint("ĐANG KHỞI TẠO MÔI TRƯỜNG H2O...")\nenv = DiagnosticRobot(\n    cfg=cfg,\n    sim_params=sim_params,\n    physics_engine=gymapi.SIM_PHYSX,\n    sim_device="cuda:0",\n    headless=True,\n)\n\ntry:\n    # Read the actual actor properties back from Isaac Gym before evaluating.\n    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])\n    actual = np.asarray([float(p.mass) for p in props])\n    original = np.asarray(env.mass_before_scaling)\n    requested = np.asarray(env.mass_requested)\n    assert (original > 0).any() and np.isfinite(actual).all()\n    assert np.allclose(requested, original * cli_args.mass_scale, rtol=1e-5, atol=1e-5)\n    assert np.allclose(actual, requested, rtol=1e-4, atol=1e-5), (actual, requested)\n    shapes = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])\n    audit = dict(\n        mass_scale=cli_args.mass_scale,\n        body_names=env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]),\n        original_mass_kg=original.tolist(), requested_mass_kg=requested.tolist(),\n        actual_mass_kg=actual.tolist(), actual_total_mass_kg=float(actual.sum()),\n        original_total_mass_kg=float(original.sum()),\n        mass_readback_verified=True,\n        inertia_recomputed_by_repository=True,\n        actor_shape_friction=[float(p.friction) for p in shapes],\n        terrain_static_friction=float(cfg.terrain.static_friction),\n        terrain_dynamic_friction=float(cfg.terrain.dynamic_friction),\n        control_dt_s=float(env.dt), seed=int(cfg.seed),\n    )\n    (output / \'physics_audit.json\').write_text(json.dumps(audit, indent=2))\n    print(\'PHYSICS VERIFIED:\', cli_args.mass_scale, \'| total mass kg:\', actual.sum())\n    obs = env.get_observations()\n    np.savez_compressed(output / \'initial_state.npz\',\n                        root=env.root_states.detach().cpu().numpy(),\n                        dof=env.dof_pos.detach().cpu().numpy(),\n                        observation=obs.detach().cpu().numpy())\n    np.savez_compressed(output / \'initial_biomechanics.npz\',\n        robot_foot_pos=env._rigid_body_pos[0, env.feet_indices].detach().cpu().numpy(),\n        foot_contact_force=env.contact_forces[0, env.feet_indices].detach().cpu().numpy(),\n        robot_dof_pos=env.dof_pos[0].detach().cpu().numpy(),\n        robot_dof_vel=env.dof_vel[0].detach().cpu().numpy(),\n        motion_start_s=env.motion_start_times.detach().cpu().numpy())\n    diagnostic_offset = (env.env_origins + env.env_origins_init_3Doffset).detach().clone()\n    diagnostic_motion_ids = env.motion_ids.detach().clone()\n    elbow_indices = [env.dof_names.index(name) for name in [\'left_elbow_joint\', \'right_elbow_joint\']]\n    holding_action = (env.dof_pos.detach().clone() - env.default_dof_pos) / cfg.control.action_scale\n    assert (holding_action[:, elbow_indices].abs() <= cfg.normalization.clip_actions).all()\n    (output / \'intervention.json\').write_text(json.dumps(dict(\n        mode=cli_args.elbow_mode, control_steps=1,\n        elbow_indices=elbow_indices, elbow_names=[env.dof_names[i] for i in elbow_indices],\n        holding_actions=holding_action[0, elbow_indices].detach().cpu().tolist(),\n        changed_commands=\'only two elbow commands at step 1; all 19 commands come from the current actor after step 1\',\n        source=\'measured dof_pos before the first simulated step; no future reference is used\',\n    ), indent=2))\n    diagnostic_motion_duration = float(env.motion_len[0].item())\n    assert abs(float(env.motion_start_times[0].item())) < 1e-7\n    assert 0 < diagnostic_motion_duration <= cli_args.max_steps * float(env.dt)\n    actor_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}\n    assert obs.shape == (1, 138)\n    print(\'CHECKPOINT:\', cli_args.checkpoint, \'| first episode only; up to\', cli_args.max_steps, \'steps\')\n    with torch.inference_mode():\n        for step in range(cli_args.max_steps):\n            assert torch.isfinite(obs).all(), \'Observation contains NaN/Inf\'\n            if cli_args.controller == \'checkpoint\':\n                actions = actor(obs)\n            elif cli_args.controller == \'zero\':\n                # Zero action means PD holds default joint targets, not zero torque.\n                actions = torch.zeros_like(env.dof_pos)\n            else:\n                assert cfg.control.control_type == \'P\'\n                assert cfg.control.action_scale > 0\n                offset = env.env_origins + env.env_origins_init_3Doffset\n                next_time = (env.episode_length_buf + 1) * env.dt + env.motion_start_times\n                reference = env._get_state_from_motionlib_cache_trimesh(\n                    env.motion_ids, next_time, offset=offset\n                )\n                actions = (reference[\'dof_pos\'] - env.default_dof_pos) / cfg.control.action_scale\n            env.diag_pd_policy_action = actions.detach().clone()\n            if cli_args.elbow_mode == \'hold_initial\' and step == 0:\n                actions = actions.clone()\n                actions[:, elbow_indices] = holding_action[:, elbow_indices]\n            clipped = torch.clamp(actions, -cfg.normalization.clip_actions,\n                                  cfg.normalization.clip_actions)\n            env.diag_joint_target = (clipped * cfg.control.action_scale\n                                     + env.default_dof_pos).detach().clone()\n            env.diag_action_abs_max = float(actions.abs().max().item())\n            env.diag_action_clip_fraction = float((actions != clipped).float().mean().item())\n            assert torch.isfinite(actions).all(), \'Action contains NaN/Inf\'\n            previous_count = len(env.diag_rows)\n            obs, _, rewards, dones, infos = env.step(actions)\n            assert len(env.diag_rows) == previous_count + 1\n            assert torch.isfinite(rewards).all(), \'Reward contains NaN/Inf\'\n            row = env.diag_rows[-1]\n            row[\'reward\'] = float(rewards[0].item())\n            if row[\'done\']:\n                print(f"Reset at frame {row[\'frame\']}, episode step {row[\'episode_step\']}: "\n                      f"{row[\'reasons\']} | contacts={row[\'contact_bodies\']} | "\n                      f"global error={row[\'global_mpjpe_mm\']:.2f} mm")\n                break\n            if row[\'motion_time_s\'] >= diagnostic_motion_duration:\n                print(\'REFERENCE MOTION END REACHED; stopping first episode\')\n                break\n\n    rows = env.diag_rows\n    assert len(env.diag_pd) == len(rows) * int(cfg.control.decimation)\n    np.savez_compressed(output / \'pd_substeps.npz\', **{\n        key: np.asarray([row[key] for row in env.diag_pd]) for key in env.diag_pd[0]})\n    (output / \'pd_metadata.json\').write_text(json.dumps(dict(\n        dof_names=list(env.dof_names), control_type=cfg.control.control_type,\n        action_scale=float(cfg.control.action_scale), clip_actions=float(cfg.normalization.clip_actions),\n        default_q=env.default_dof_pos[0].detach().cpu().tolist(),\n        p_gains=env.p_gains.detach().cpu().tolist(), d_gains=env.d_gains.detach().cpu().tolist(),\n        torque_limits_Nm=env.torque_limits.detach().cpu().tolist(),\n        decimation=int(cfg.control.decimation), control_dt_s=float(env.dt),\n        simulation_dt_s=float(env.sim_params.dt),\n        note=\'Each substep snapshot is before simulation integration. Returned torque is the original controller output. \'\n             \'A normalized action is an offset from default_q, not an absolute angle.\',\n    ), indent=2))\n    assert len(env.diag_biomechanics) == len(rows)\n    np.savez_compressed(output / \'biomechanics_before_reset.npz\', **{\n        key: np.asarray([row[key] for row in env.diag_biomechanics])\n        for key in env.diag_biomechanics[0]})\n    for name, events in env.diag_events.items():\n        assert len(events) == len(rows), (name, len(events), len(rows))\n        np.savez_compressed(output / (\'reward_event_\' + name + \'.npz\'), **{\n            key: np.asarray([row[key] for row in events]) for key in events[0]})\n    (output / \'biomechanics_metadata.json\').write_text(json.dumps(dict(\n        foot_body_indices=env.feet_indices.detach().cpu().tolist(),\n        foot_body_names=[env._body_list[int(i)] for i in env.feet_indices],\n        dof_names=list(env.dof_names),\n        velocity_limits_rad_s=env.dof_vel_limits.detach().cpu().tolist(),\n        soft_velocity_limit_factor=float(cfg.rewards.soft_dof_vel_limit),\n        desired_foot_max_height_m=float(cfg.rewards.desired_feet_max_height_for_this_air),\n        contact_force_threshold_N=1.0,\n        position_scope=\'world z of rigid-body origins; not sole clearance above terrain\',\n        reference_contact_force_available=False,\n        timing=\'snapshots before reset; reward-event snapshots taken at the original function call\',\n    ), indent=2))\n    # Query the complete reference only AFTER all rollout samples have been copied.\n    motion_id = int(diagnostic_motion_ids[0].item())\n    count = int(env._motion_lib._motion_num_frames[motion_id].item())\n    frame_dt = float(env._motion_lib._motion_dt[motion_id].item())\n    profile = []\n    for index in range(count):\n        time = torch.full_like(env.motion_start_times, index * frame_dt)\n        state = env._get_state_from_motionlib_cache_trimesh(\n            diagnostic_motion_ids, time, offset=diagnostic_offset)\n        profile.append(dict(time_s=index * frame_dt,\n            foot_pos=state[\'rg_pos\'][0, env.feet_indices].detach().cpu().numpy().copy(),\n            dof_pos=state[\'dof_pos\'][0].detach().cpu().numpy().copy(),\n            dof_vel=state[\'dof_vel\'][0].detach().cpu().numpy().copy()))\n    np.savez_compressed(output / \'reference_full_motion.npz\', **{\n        key: np.asarray([sample[key] for sample in profile]) for key in profile[0]})\n    assert len(env.diag_reward_rows) == len(rows)\n    with (output / \'reward_terms.csv\').open(\'w\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(env.diag_reward_rows[0]))\n        writer.writeheader()\n        writer.writerows(env.diag_reward_rows)\n    (output / \'reward_scale_audit.json\').write_text(json.dumps(dict(\n        runtime_scales={key: float(value) for key, value in env.reward_scales.items()},\n        penalty_scale=float(cfg.rewards.penalty_scale),\n        penalty_reward_names=list(cfg.rewards.penalty_reward_names),\n        only_positive_rewards=bool(cfg.rewards.only_positive_rewards),\n        contribution_method=\'episode_sums delta around one original compute_reward call, before reset; float32 counter quantization possible\',\n        total_reconstruction_atol=1e-3, total_reconstruction_rtol=1e-4,\n    ), indent=2))\n    assert 1 <= len(rows) <= cli_args.max_steps\n    assert all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())\n    with (output / \'frames.csv\').open(\'w\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))\n        writer.writeheader()\n        writer.writerows(rows)\n    np.savez_compressed(output / \'positions_before_reset.npz\',\n                        robot=np.stack(env.diag_pred), reference=np.stack(env.diag_ref))\n    reason_counts = Counter(reason for row in rows if row[\'done\']\n                            for reason in row[\'reasons\'].split(\';\') if reason)\n    summary = dict(\n        purpose=\'first-step elbow target intervention, then original actor until first termination or motion end; not paper evaluation\',\n        training_seed=int(metadata[\'seed\']),\n        evaluation_seed=1,\n        intervention_mode=cli_args.elbow_mode,\n        mass_scale=cli_args.mass_scale,\n        actor_weights_unchanged=True,\n        maximum_control_steps=cli_args.max_steps,\n        controller=cli_args.controller,\n        checkpoint_name=cli_args.checkpoint,\n        checkpoint_used_for_actions=(cli_args.controller == \'checkpoint\'),\n        max_requested_action=float(max(row[\'requested_action_abs_max\'] for row in rows)),\n        mean_action_clip_fraction=float(np.mean([row[\'action_clip_fraction\'] for row in rows])),\n        mean_torque_limit_fraction=float(np.mean([row[\'torque_limit_fraction\'] for row in rows])),\n        metric_scope=\'first episode only, including terminal frame; compare MPJPE on common nonterminal prefix\',\n        checkpoint_provenance=\'user fresh PPO run; archived hyperparameter source provenance unknown\',\n        training_run=folder.name,\n        training_initialization=metadata[\'initialization\'],\n        deterministic_actor=True,\n        checkpoint_sha256=hashlib.sha256((folder / cli_args.checkpoint).read_bytes()).hexdigest(),\n        seed=cfg.seed,\n        control_dt_s=float(env.dt),\n        frame_count=len(rows),\n        reset_count=sum(row[\'done\'] for row in rows),\n        first_reset_step=next((r[\'episode_step\'] for r in rows if r[\'done\']), None),\n        first_reset_time_s=next((r[\'motion_time_s\'] for r in rows if r[\'done\']), None),\n        reset_reason_counts=dict(reason_counts),\n        observed_frame_global_mpjpe_mm=float(np.mean([r[\'global_mpjpe_mm\'] for r in rows])),\n        observed_frame_root_relative_mpjpe_mm=float(np.mean([r[\'root_relative_mpjpe_mm\'] for r in rows])),\n        num_body_points=int(env.diag_pred[0].shape[0]),\n        reference_distance_termination_threshold_m=float(cfg.asset.termination_scales.max_ref_motion_distance),\n        success_rate=None,\n        note=\'Multiple causes may occur in one reset; success rate is not measured here.\',\n    )\n    last = rows[-1]\n    failure = bool(last[\'done\']) and any(reason != \'motion_timeout\' for reason in last[\'reasons\'].split(\';\') if reason)\n    reached = last[\'motion_time_s\'] >= diagnostic_motion_duration\n    summary.update(motion_duration_s=diagnostic_motion_duration, intervention_control_steps=1,\n        failure_time_s=last[\'motion_time_s\'] if failure else None,\n        reached_motion_end_without_recorded_failure=bool(reached and not failure),\n        stop_reason=\'termination\' if failure else (\'motion_end\' if reached else \'step_limit\'))\n    (output / \'summary.json\').write_text(json.dumps(summary, indent=2))\n    print(\'\\nDIAGNOSTIC SUMMARY:\')\n    print(json.dumps(summary, indent=2))\n    print(\'CHAN DOAN RESET HOAN TAT\')\n    print(\'Results:\', output)\nfinally:\n    env.gym.destroy_sim(env.sim)\n'
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile

CASES = [(seed, stage, filename) for seed in [1, 2, 3]
         for stage, filename in [('initial', 'model_initial.pt'), ('final', 'model_200.pt')]]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def capture(command, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    return result.stdout

def lines(text):
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')}

def summarize(output, source_hashes):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def arrays(path):
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}

    def read_frames(folder):
        with (folder / 'frames.csv').open() as stream:
            return list(csv.DictReader(stream))

    def prefix_length(frames, duration):
        count = 0
        for row in frames:
            if row['done'] == 'True' or float(row['motion_time_s']) >= duration:
                break
            count += 1
        return count

    def errors(positions, count):
        robot = positions['robot'][:count].astype(np.float64)
        reference = positions['reference'][:count].astype(np.float64)
        global_values = np.linalg.norm(robot - reference, axis=-1).mean(axis=1) * 1000
        relative_values = np.linalg.norm(
            (robot - robot[:, [0]]) - (reference - reference[:, [0]]), axis=-1).mean(axis=1) * 1000
        return dict(global_mpjpe_mm=float(global_values.mean()),
                    root_relative_mpjpe_mm=float(relative_values.mean()),
                    maximum_frame_global_mpjpe_mm=float(global_values.max()))

    data = {}
    audits = []
    all_configs, all_physics = [], []
    max_torque_difference = 0.0
    for seed, stage, filename in CASES:
        case = 'seed%d_%s' % (seed, stage)
        for mode in ['original', 'hold_initial']:
            folder = output / (case + '_' + mode)
            summary = load(folder / 'summary.json')
            meta = load(folder / 'pd_metadata.json')
            intervention = load(folder / 'intervention.json')
            initial = arrays(folder / 'initial_state.npz')
            initial_bio = arrays(folder / 'initial_biomechanics.npz')
            positions = arrays(folder / 'positions_before_reset.npz')
            bio = arrays(folder / 'biomechanics_before_reset.npz')
            pd = arrays(folder / 'pd_substeps.npz')
            frames = read_frames(folder)
            duration = float(summary['motion_duration_s'])
            assert summary['training_seed'] == seed and summary['evaluation_seed'] == 1
            assert summary['checkpoint_sha256'] == source_hashes[case]
            assert summary['actor_weights_unchanged'] and summary['intervention_mode'] == mode
            assert summary['maximum_control_steps'] == 300
            assert 1 <= len(frames) == summary['frame_count'] <= 300
            assert all(int(row['episode']) == 0 for row in frames)
            assert all(row['done'] == 'False' for row in frames[:-1])
            assert [int(row['episode_step']) for row in frames] == list(range(1, len(frames) + 1))
            assert positions['robot'].shape == positions['reference'].shape == (len(frames), 22, 3)
            assert meta['decimation'] == 4 and meta['control_type'] == 'P'
            assert pd['q'].shape == (4 * len(frames), 19)
            assert np.array_equal(pd['control_step'], np.repeat(np.arange(1, len(frames)+1), 4))
            assert np.array_equal(pd['substep'], np.tile([1, 2, 3, 4], len(frames)))
            assert all(np.isfinite(value).all() for mapping in [initial, positions, pd]
                       for value in mapping.values())
            assert np.allclose(pd['q'][0], initial_bio['robot_dof_pos'], rtol=0, atol=1e-6)
            ids = intervention['elbow_indices']
            assert [meta['dof_names'][j] for j in ids] == ['left_elbow_joint', 'right_elbow_joint']
            command = pd['policy_raw_action'].copy()
            if mode == 'hold_initial':
                command[:4, ids] = ((initial_bio['robot_dof_pos'][ids]
                                    - np.array(meta['default_q'])[ids]) / meta['action_scale'])
            applied = np.clip(command, -meta['clip_actions'], meta['clip_actions'])
            assert np.allclose(applied, pd['torque_input_action'], rtol=1e-5, atol=1e-6)
            # After the first control step every command must come from the current actor.
            if len(frames) > 1:
                assert np.allclose(pd['torque_input_action'][4:], np.clip(
                    pd['policy_raw_action'][4:], -meta['clip_actions'], meta['clip_actions']),
                    rtol=1e-5, atol=1e-6)
            target = applied * meta['action_scale'] + np.array(meta['default_q'])
            proportional = pd['kp_scale'] * np.array(meta['p_gains']) * (target-pd['q'])
            damping = -pd['kd_scale'] * np.array(meta['d_gains']) * pd['qdot']
            torque = proportional + damping
            limits = np.array(meta['torque_limits_Nm'])
            reconstructed = np.clip(torque, -limits, limits)
            max_torque_difference = max(max_torque_difference, float(
                np.abs(reconstructed-pd['returned_torque']).max()))
            for expected, measured in [(target, pd['target_q']),
                                       (proportional, pd['proportional_torque']),
                                       (damping, pd['damping_torque']),
                                       (torque, pd['unclipped_torque']),
                                       (reconstructed, pd['returned_torque'])]:
                assert np.allclose(expected, measured, rtol=1e-5, atol=1e-4)
            if mode == 'hold_initial':
                assert np.allclose(target[:4, ids], initial_bio['robot_dof_pos'][ids],
                                   rtol=0, atol=1e-6)
            # Audit metrics from saved world positions, not from text logs alone.
            recomputed = np.linalg.norm(positions['robot'].astype(float)
                                       - positions['reference'].astype(float), axis=-1).mean(axis=1)*1000
            reported = np.asarray([float(row['global_mpjpe_mm']) for row in frames])
            assert np.allclose(recomputed, reported, rtol=1e-5, atol=1e-3)
            assert np.isclose(float(frames[-1]['motion_time_s']),
                              len(frames)*meta['control_dt_s'], rtol=0, atol=2e-6)
            all_configs.append(load(folder / 'config_requested.json'))
            all_physics.append(load(folder / 'physics_audit.json'))
            data[(seed, stage, mode)] = dict(folder=folder, summary=summary, meta=meta,
                initial=initial, initial_bio=initial_bio, positions=positions,
                bio=bio, pd=pd, frames=frames, duration=duration, ids=ids,
                torque=torque, limits=limits, prefix=prefix_length(frames, duration))

    assert all(value == all_configs[0] for value in all_configs)
    assert all(value == all_physics[0] for value in all_physics)
    common_steps = min(value['prefix'] for value in data.values())
    assert common_steps > 0, 'No shared nonterminal prefix; do not compute a misleading MPJPE'
    records, changes, elbow_records = [], [], []
    for seed, stage, filename in CASES:
        original = data[(seed, stage, 'original')]
        held = data[(seed, stage, 'hold_initial')]
        case = 'seed%d_%s' % (seed, stage)
        assert original['meta'] == held['meta']
        for key in ['root', 'dof', 'observation']:
            delta = np.abs(original['initial'][key].astype(float)-held['initial'][key].astype(float))
            tolerance = np.full(delta.shape, 1e-6)
            if key == 'observation':
                tolerance[:, 41:44] = 2e-6
            audits.append(dict(case=case, field=key, max_absolute_difference=float(delta.max()),
                               passed=bool(np.all(delta <= tolerance))))
        assert np.allclose(original['pd']['policy_raw_action'][:4],
                           held['pd']['policy_raw_action'][:4], rtol=0, atol=1e-6)
        other = [j for j in range(19) if j not in original['ids']]
        assert np.allclose(original['pd']['torque_input_action'][:4, other],
                           held['pd']['torque_input_action'][:4, other], rtol=0, atol=1e-6)
        pair_steps = min(original['prefix'], held['prefix'])
        pair_records = []
        for mode in ['original', 'hold_initial']:
            value = data[(seed, stage, mode)]
            summary = value['summary']
            full_prefix = errors(value['positions'], value['prefix'])
            metrics = errors(value['positions'], common_steps)
            pair_metrics = errors(value['positions'], pair_steps)
            record = dict(training_seed=seed, stage=stage, mode=mode,
                observed_steps=len(value['frames']),
                observed_duration_s=float(value['frames'][-1]['motion_time_s']),
                motion_duration_s=value['duration'], stop_reason=summary['stop_reason'],
                reached_motion_end_without_recorded_failure=summary['reached_motion_end_without_recorded_failure'],
                first_reset_time_s=summary['first_reset_time_s'],
                failure_time_s=summary['failure_time_s'],
                terminal_reasons=value['frames'][-1]['reasons'],
                common_nonterminal_steps=common_steps,
                common_duration_s=common_steps*value['meta']['control_dt_s'],
                common_global_mpjpe_mm=metrics['global_mpjpe_mm'],
                common_root_relative_mpjpe_mm=metrics['root_relative_mpjpe_mm'],
                common_maximum_frame_global_mpjpe_mm=metrics['maximum_frame_global_mpjpe_mm'],
                pair_common_nonterminal_steps=pair_steps,
                pair_common_global_mpjpe_mm=pair_metrics['global_mpjpe_mm'],
                own_nonterminal_steps=value['prefix'],
                own_prefix_global_mpjpe_mm=full_prefix['global_mpjpe_mm'],
                own_prefix_maximum_frame_global_mpjpe_mm=full_prefix['maximum_frame_global_mpjpe_mm'])
            records.append(record)
            pair_records.append(record)
            for j in value['ids']:
                clipped = np.abs(value['torque'][:, j]) > value['limits'][j]+1e-5
                elbow_records.append(dict(training_seed=seed, stage=stage, mode=mode,
                    joint=value['meta']['dof_names'][j],
                    first_step_abs_velocity_rad_s=float(abs(value['bio']['robot_dof_vel'][0, j])),
                    second_step_abs_velocity_rad_s=(float(abs(value['bio']['robot_dof_vel'][1, j]))
                                                    if len(value['frames']) > 1 else None),
                    first_step_clipped_substeps=int(clipped[:4].sum()),
                    second_step_clipped_substeps=(int(clipped[4:8].sum()) if len(value['frames'])>1 else None),
                    own_prefix_peak_abs_velocity_rad_s=float(np.abs(
                        value['bio']['robot_dof_vel'][:, j]).max())))
        before, after = pair_records
        changes.append(dict(training_seed=seed, stage=stage,
            observed_duration_original_s=before['observed_duration_s'],
            observed_duration_hold_first_step_s=after['observed_duration_s'],
            observed_duration_change_s=after['observed_duration_s']-before['observed_duration_s'],
            original_reached_motion_end=before['reached_motion_end_without_recorded_failure'],
            hold_first_step_reached_motion_end=after['reached_motion_end_without_recorded_failure'],
            common_global_change_mm=after['common_global_mpjpe_mm']-before['common_global_mpjpe_mm'],
            common_root_relative_change_mm=after['common_root_relative_mpjpe_mm']-before['common_root_relative_mpjpe_mm']))
    assert all(row['passed'] for row in audits), 'Paired initial states differ; inspect saved data'
    write(output / 'paired_initial_state_audit.json', dict(checks=audits, rtol=0,
        default_atol=1e-6, angular_velocity_observation_atol=2e-6))
    write(output / 'elbow_episode_comparison.json', dict(
        purpose='paired first-step elbow intervention followed by the original actor until termination or motion end',
        common_nonterminal_steps=common_steps, results=records, paired_changes=changes,
        elbow_transients=elbow_records, training_performed=False, actor_weights_unchanged=True,
        first_step_other_17_commands_matched=True, physical_parameters_and_configs_equal=True,
        maximum_torque_reconstruction_difference_Nm=max_torque_difference, success_rate=None,
        notes=['Only the first control step is overridden; later actor commands can differ due to changed observations.',
               'Own-prefix errors and peak velocities span different durations and are descriptive, not matched comparisons.',
               'Reaching the reference end under local termination thresholds is not the paper success-rate protocol.',
               'At motion end observed duration is censored; do not treat it as measured time to failure.',
               'This diagnostic intervention is not a retrained or validated improved learned policy.']))
    for name, rows in [('episode_results.csv', records), ('paired_changes.csv', changes),
                       ('elbow_transients.csv', elbow_records)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ['s%d_%s' % (seed, stage) for seed, stage, _ in CASES]
    x = np.arange(len(CASES))
    for mode, offset, label in [('original', -.18, 'Original actor'),
                                ('hold_initial', .18, 'Hold elbows only at step 1')]:
        selected = [row for row in records if row['mode'] == mode]
        axes[0,0].bar(x+offset, [row['observed_duration_s'] for row in selected], width=.36, label=label)
        axes[0,1].bar(x+offset, [row['common_global_mpjpe_mm'] for row in selected], width=.36, label=label)
        value = data[(1, 'final', mode)]
        times = [float(row['motion_time_s']) for row in value['frames']]
        axes[1,0].plot(times, [float(row['global_mpjpe_mm']) for row in value['frames']], label=label)
        axes[1,1].plot(times, [float(row['gravity_x']) for row in value['frames']], label=label)
    for ax in axes[0]:
        ax.set_xticks(x);ax.set_xticklabels(labels, rotation=25)
    axes[0,0].set(title='Observed duration; motion-end cases are censored', ylabel='Seconds')
    axes[0,1].set(title='Global MPJPE on shared %d nonterminal steps' % common_steps, ylabel='mm')
    axes[1,0].set(title='Seed 1 final: includes terminal sample', xlabel='Motion time (s)', ylabel='Global MPJPE (mm)')
    axes[1,1].set(title='Seed 1 final: torso gravity X', xlabel='Motion time (s)')
    axes[1,1].axhline(.7, color='gray', ls='--');axes[1,1].axhline(-.7, color='gray', ls='--')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Paired first-step elbow intervention, then actor control | no training')
    fig.savefig(output / 'elbow_episode_comparison.png', dpi=180);plt.close(fig)
    print('\nPAIRED EPISODE RESULTS | shared nonterminal steps:', common_steps, flush=True)
    for change in changes:
        print('seed%d_%s: %.2f -> %.2f s | common global change=%+.3f mm' % (
            change['training_seed'], change['stage'], change['observed_duration_original_s'],
            change['observed_duration_hold_first_step_s'], change['common_global_change_mm']), flush=True)


def main():
    root = Path('/content/hr17_elbow_episode')
    output = root / datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    python = Path('/content/hr17_env/bin/python')
    repo = Path('/content/hr17_repo')
    status = dict(status='preparing', training_seeds=[1, 2, 3], evaluation_seed=1,
                  checkpoints=['model_initial.pt', 'model_200.pt'], training_performed=False,
                  maximum_control_steps_per_case=300, intervention_control_steps=1, intervention_modes=['original', 'hold_initial'])
    try:
        assert python.is_file() and repo.is_dir(), 'Restore the environment first'
        pointer = Path('/content/hr17_three_seeds/latest_archive.txt')
        assert pointer.is_file(), 'Missing three-seed training index'
        group = Path(pointer.read_text().strip()).with_suffix('')
        training_status = load(group / 'experiment_status.json')
        assert training_status['status'] == 'completed'
        runs = {int(seed): Path(path) for seed, path in training_status['runs'].items()}
        assert sorted(runs) == [1, 2, 3]
        baseline = runs[1]
        first_metadata = load(baseline / 'run_metadata.json')
        base_cfg = load(baseline / 'config_requested.json')
        import copy
        source_hashes = {}
        for seed, run in sorted(runs.items()):
            assert run.is_dir(), 'Restore run directory: ' + str(run)
            metadata = load(run / 'run_metadata.json')
            assert metadata['status'] == 'completed' and metadata['seed'] == seed
            assert metadata['iterations'] == 200 and metadata['num_envs'] == 128
            assert sha(run / 'model_200.pt') == metadata['final_checkpoint_sha256']
            cfg = load(run / 'config_requested.json')
            expected = copy.deepcopy(base_cfg); expected['seed'] = seed; expected['train']['seed'] = seed
            assert cfg == expected
            for stage, filename in [('initial', 'model_initial.pt'), ('final', 'model_200.pt')]:
                source_hashes['seed%d_%s' % (seed, stage)] = sha(run / filename)
        robot_source = repo / 'legged_gym/legged_gym/envs/base/legged_robot.py'
        before_source_hash = sha(robot_source)
        assert 'quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.clear)' not in robot_source.read_text(), 'Run gravity fix cell 18 first'
        assert capture(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).strip() == first_metadata['repo_commit']
        diff = capture(['git', 'diff', '--no-ext-diff'], cwd=str(repo))
        assert diff.splitlines() == (baseline / 'repo_changes.patch').read_text().splitlines(), 'Repo differs from training'
        env = os.environ.copy()
        env.update(MAX_JOBS='2', MPLBACKEND='Agg', PYTHONHASHSEED='1', PYTHONUNBUFFERED='1')
        env['PATH'] = '/content/hr17_env/bin:' + env.get('PATH', '')
        runtime = load(Path('/content/hr17_runtime_environment.json'))
        assert runtime.get('native_h2o_import_verified'), 'Complete the direct H2O import check first'
        env['LD_LIBRARY_PATH'] = runtime['LD_LIBRARY_PATH']
        write(output / 'runtime_environment.json', runtime)
        freeze = capture([str(python), '-m', 'pip', 'freeze'], env=env)
        assert lines(freeze) == lines((baseline / 'environment_freeze.txt').read_text()), 'Environment differs from training'
        (output / 'environment_freeze.txt').write_text(freeze)
        (output / 'repo_diff.patch').write_text(diff)
        (output / 'gpu_info.txt').write_text(capture(['nvidia-smi']))
        shutil.copy2(group / 'run_index.json', output / 'training_run_index.json')
        shutil.copy2(group / 'training_comparison.json', output / 'source_training_comparison.json')
        shutil.copy2(__file__, output / Path(__file__).name)
        child = output / 'evaluate_case.py'
        child.write_text(DIAGNOSTIC_SOURCE, encoding='utf-8')
        write(output / 'source_checkpoint_hashes.json', source_hashes)
        status['status'] = 'evaluating'
        for seed, stage, filename in CASES:
            for mode in ['original', 'hold_initial']:
                label = 'seed%d_%s_%s' % (seed, stage, mode)
                print('\nPAIRED EPISODE TEST:', label, flush=True)
                command = [str(python), '-u', str(child), '--checkpoint', filename,
                    '--run-dir', str(runs[seed]), '--output', str(output / label),
                    '--max-steps', '300', '--elbow-mode', mode]
                print('Each rollout has a 180-second wall-clock limit; checkpoint weights stay unchanged.', flush=True)
                try:
                    result = subprocess.run(command, cwd=str(repo), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
                    code, log_text = result.returncode, result.stdout
                except subprocess.TimeoutExpired as error:
                    log_text = error.stdout or b''
                    if isinstance(log_text, bytes): log_text = log_text.decode(errors='replace')
                    code, log_text = 124, log_text + '\nTIMEOUT after 180 seconds\n'
                (output / (label + '.log')).write_text(log_text, encoding='utf-8')
                print(log_text[-3000:], flush=True)
                assert code == 0, label + ' failed; see saved log'
        summarize(output, source_hashes)
        for seed, stage, filename in CASES:
            assert sha(runs[seed] / filename) == source_hashes['seed%d_%s' % (seed, stage)]
        assert sha(robot_source) == before_source_hash
        status.update(status='completed', source_checkpoint_files_unchanged=True, repository_source_unchanged=True)
        print('\nELBOW EPISODE COMPARISON COMPLETED', flush=True)
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(output / 'experiment_status.json', status)
        manifest = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob('*') if path.is_file()}
        write(output / 'SHA256_manifest.json', manifest)
        archive = shutil.make_archive(str(output), 'zip', root_dir=str(output))
        with zipfile.ZipFile(archive) as package: assert package.testzip() is None
        (root / 'latest_archive.txt').write_text(archive)
        print('RESULT ZIP:', archive, flush=True)

if __name__ == '__main__':
    main()
