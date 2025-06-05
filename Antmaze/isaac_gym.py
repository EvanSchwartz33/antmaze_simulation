from isaacgym import gymapi, gymutil, gymtorch
import torch
import numpy as np


gym = gymapi.acquire_gym()
args = gymutil.parse_arguments()


sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.substeps = 2
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")


viewer = gym.create_viewer(sim, gymapi.CameraProperties())


asset_root = "assets"
ant_asset = gym.load_asset(sim, asset_root, "urdf/ant/ant.urdf", gymapi.AssetOptions())


env_spacing = 3.0
env = gym.create_env(sim, gymapi.Vec3(-env_spacing, -env_spacing, 0.0),
                          gymapi.Vec3(env_spacing, env_spacing, 0.0), 1)


ant_pose = gymapi.Transform()
ant_pose.p = gymapi.Vec3(0.0, 0.0, 0.4)
ant_actor = gym.create_actor(env, ant_asset, ant_pose, "ant", 0, 1)


box_asset = gym.create_box(sim, 0.2, 2.0, 1.0, gymapi.AssetOptions())
wall_positions = [
    (1.5, 0.0), 
    (-1.5, 0.0), 
    (0.0, 1.5),  
    (0.0, -1.5), 
]

for i, (x, y) in enumerate(wall_positions):
    wall_pose = gymapi.Transform()
    wall_pose.p = gymapi.Vec3(x, y, 0.5)
    gym.create_actor(env, box_asset, wall_pose, f"wall{i}", i + 1, 0)


dof_props = gym.get_actor_dof_properties(env, ant_actor)
num_dofs = len(dof_props)


gym.prepare_sim(sim)
obs_tensor = gym.acquire_dof_state_tensor(sim)
obs = gymtorch.wrap_tensor(obs_tensor)


while not gym.query_viewer_has_closed(viewer):
   
    gym.simulate(sim)
    gym.fetch_results(sim, True)


    for i in range(num_dofs):
        gym.set_actor_dof_position_target(env, ant_actor, i, np.random.uniform(-0.5, 0.5))

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
