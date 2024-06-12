import torch
import torch.nn as nn
import math
from scene.sky.utils import get_ray_directions

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions, include_input=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input

    def forward(self, x):
        # x.shape = [batch_size, num_samples, 3]
        batch_size, num_samples, _ = x.shape

        # Create a tensor of encoding functions
        encoding_functions = torch.arange(0, self.num_encoding_functions, dtype=torch.float32, device=x.device)
        encoding_functions = 2 ** (encoding_functions) * math.pi  # [num_encoding_functions]

        # Encode the input tensor
        encoded_inputs = x.unsqueeze(-1) * encoding_functions.view(1, 1, 1, -1)  # [batch_size, num_samples, 3, num_encoding_functions]
        encoded_inputs = torch.cat([torch.sin(encoded_inputs), torch.cos(encoded_inputs)], dim=-1)  # [batch_size, num_samples, 3, 2 * num_encoding_functions]
        encoded_inputs = encoded_inputs.view(batch_size, num_samples, -1)  # [batch_size, num_samples, 3 * 2 * num_encoding_functions]

        if self.include_input:
            encoded_inputs = torch.cat([x, encoded_inputs], dim=-1)  # [batch_size, num_samples, 3 + 3 * 2 * num_encoding_functions]

        return encoded_inputs

class SkyMlp(nn.Module):
    def __init__(self, sky_model_args):
        super(SkyMlp, self).__init__()

        num_encoding_functions = sky_model_args.num_encoding_functions
        hidden_dim = sky_model_args.hidden_dim
        
        self.positional_encoding = PositionalEncoding(num_encoding_functions)
        self.fc1 = nn.Linear(3 + 3 * 2 * num_encoding_functions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()
    
    def capture(self):
        return self.state_dict()
    
    def train_params(self):
        return self.parameters()
    
    def restore(self, model_args):
        self.load_state_dict(model_args)

    def _forward(self, view_dir):
        """
        Input:
            view_dir: torch.Tensor of shape [batch_size, num_samples, 3]
        Returns:
            rgb: torch.Tensor of shape [batch_size, num_samples, 3]
        """
        x = self.positional_encoding(view_dir)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def forward(self, viewpoint_camera):
        c2w = torch.linalg.inv(viewpoint_camera.world_view_transform.transpose(0, 1))
        ray_d_world = get_ray_directions(viewpoint_camera.image_height, 
                                         viewpoint_camera.image_width, 
                                         viewpoint_camera.FoVx, 
                                         viewpoint_camera.FoVy, 
                                         c2w).cuda()  # [H, W, 3]
        
        ray_d_world_batch = ray_d_world.view(1, -1, 3)

        skymap = self._forward(ray_d_world_batch).view(viewpoint_camera.image_height, viewpoint_camera.image_width, 3).permute(2, 0, 1) # [3, H, W]

        return skymap