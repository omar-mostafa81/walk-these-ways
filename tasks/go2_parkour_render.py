import numpy as np
import os, time
import pickle

from isaacgym import gymapi
from isaacgym import gymtorch

import torch
import torchvision

from tasks.go2_parkour import Go2Parkour

class Go2ParkourRender(Go2Parkour):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_buffer = []

    def create_sim(self):
        super().create_sim()
        self.camera_height = 360 #540
        self.camera_width = 480 #960

        self.camera_offsets = [
                [0.3, -0.3, 0.1, 0.0, 0.0, -0.1],
                [0.4, 0.0, -0.05, 0.0, 0.0, -0.05],
                [0.0, 0.5, -0.05, 0.0, 0.0, -0.05],
                [0.3, 0.3, 1.0, 0.0, 0.0, 0.0],
        ]
        self.camera_handles = []
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_width
        camera_props.height = self.camera_height
        camera_props.enable_tensors = True
        for _ in self.camera_offsets:
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            assert camera_handle != -1, "The camera failed to be created"
            self.camera_handles.append(camera_handle)

    def write_video(self, video_frames, filepath, fps=50):
        if True:
            torchvision.io.write_video(filepath, video_frames, fps=fps, video_codec='libx264', options={'crf': '18'})
            return

        import cv2
        video_frames_np = video_frames.cpu().numpy() # Assuming you're using a CPU tensor
        height, width, _ = video_frames_np[0].shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height), isColor=True)

        for frame in video_frames_np:
            out.write(frame)
        out.release()

    def compute_constraints_cat(self):
        # Constraint to avoid having the robot upside-down
        cstr_upsidedown = self.projected_gravity[:, 2] > 0
        cstr_termination_contacts = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        cstr_lava = self.root_states[:, 2] < -0.05
        cstr_minbaseheight = (self.limits["min_base_height"] - self.root_states[:, 2]) * (self.ceilings >= 0.34).float()
        cstr_base_height_min = torch.any(cstr_minbaseheight > 0)

        self.cstr_manager.add("upsidedown", cstr_upsidedown, max_p=1.0)
        self.cstr_manager.log_all(self.episode_sums)

        # Timeout if episodes have reached their maximum duration
        timeout = self.progress_buf >= self.max_episode_length - 1

        # Get final termination probability for each env from all constraints
        self.cstr_prob = self.cstr_manager.get_probs()

        # Probability of termination used to affect the discounted sum of rewards
        self.reset_buf = self.cstr_prob
        self.power = self.reset_buf

        self.reset_env_buf = timeout | cstr_upsidedown | cstr_lava | cstr_termination_contacts | cstr_base_height_min
        if self.reset_env_buf.any():
            print(
                f"cstr_upsidedown: {cstr_upsidedown.any()}, cstr_termination_contacts: {cstr_termination_contacts.any()}, "
                f"cstr_lava: {cstr_lava.any()}, cstr_base_height_min: {cstr_base_height_min.any()}"
            )
            print(f"cstr_termination_contacts: {torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.}")




    def post_physics_step(self):
        super().post_physics_step()

        ###########################################################################
        env_ids = slice(0,1)
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # If robots went far enough during their episode, move them up.
        # This is monitored by move_up_flag that is set to True during the episode.
        # If robots did not move enough with respect to their velocity command, move them down.
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~self.move_up_flag[env_ids] # <==========================replace 0.25 by 0.33
        move_up = distance > self.terrain.env_length * 0.8
        move_down = distance < self.terrain.env_length * 0.5 
        terrain_levels = self.terrain_levels.clone()
        terrain_levels[env_ids] = 1 * move_up - 1 * move_down
        #self.terrain_levels[env_ids] += 1 * self.move_up_flag[env_ids] - 1 * move_down
        #self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        terrain_levels[env_ids] = torch.where(
                terrain_levels[env_ids]>=self.terrain.env_rows,
                torch.randint_like(terrain_levels[env_ids], self.terrain.env_rows),
                torch.clip(terrain_levels[env_ids], 0)
        )
        print(f"distance: {distance}, move_up: {move_up}, move_down: {move_down}, terrain_levels: {terrain_levels}")
        ###########################################################################

        # Sample half of the time
        if self.common_step_counter % 1 == 0: 
            # Manually put the camera at the desired location, as in PA's code
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            base_pos = (self.root_states[0, :3]).cpu().numpy()
            for i, camera_offset in enumerate(self.camera_offsets):
                cam_pos = gymapi.Vec3(base_pos[0] + camera_offset[0], base_pos[1] + camera_offset[1], base_pos[2] + camera_offset[2])
                cam_target = gymapi.Vec3(base_pos[0] + camera_offset[3], base_pos[1] + camera_offset[4], base_pos[2] + camera_offset[5])
                self.gym.set_camera_location(self.camera_handles[i], self.envs[0], cam_pos, cam_target)
    
            # Render simulation
            self.gym.step_graphics(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)
    
            # Access camera image
            self.gym.start_access_image_tensors(self.sim)
            images = []
            for i in range(len(self.camera_offsets)):
                image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.camera_handles[i], gymapi.IMAGE_COLOR)
                image = gymtorch.wrap_tensor(image_)
                images.append(image)
            self.gym.end_access_image_tensors(self.sim)
    
            # Store image in replay buffer
            images = [image.view(self.camera_height, self.camera_width, 4)[:, :, :3].cpu() for image in images]
            if len(images) < 3:
                frame = torch.hstack(images)
            elif len(images) == 4:
                frame = torch.vstack([torch.hstack(images[:2]), torch.hstack(images[2:])])
            else:
                assert False, "Number of cameras not supported (yet)"
            self.video_buffer.append(frame)

        if self.reset_env_buf[0] == 1.0 and len(self.video_buffer) > 0:

            print("====================")
            save_path = self.cfg['video_save_path']
            print(f"Saving {save_path} and reseting video_buffer")
            video = torch.stack(self.video_buffer, dim=0)
            self.write_video(video, save_path, fps=50.0)
            #torchvision.io.write_video(save_path, video, fps=50.0)
            print("video saved")
            self.video_buffer = []
            exit()



