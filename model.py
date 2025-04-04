import torch
import torch.nn as nn

from layers.common_layers import build_mlp
from layers.transformer_encoder_layer import TransformerEncoderLayer
from modules.agent_encoder import AgentEncoder
from modules.map_encoder import MapEncoder
from modules.trajectory_decoder import TrajectoryDecoder

# no meaning, required by nuplan


class PlanningModel(nn.Module):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=4,
        history_channel=9,
        history_steps=10,
        future_steps=30,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=True,
        state_attn_encoder=True,
        state_dropout=0.75,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, data, **kwargs):
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_eval(data)

    def forward_model(self, data):
        x_agent = self.agent_encoder(data) #[B, N_obj, dim] Core vector to represent ego + objects
        x_polygon = self.map_encoder(data) #[B, N_map, dim] Core vector to represent map

        bs, A, _ = x_agent.shape # if use_ego_history, it will include ego

        x = torch.cat([x_agent, x_polygon], dim=1) # []

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x)

        trajectory, probability = self.trajectory_decoder(x[:, 0])
        #prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        probability = probability.softmax(dim=-1)
        out = {
            "trajectory": trajectory, #[B, 6, 30, 4], x, y, dy,dx
            "probability": probability, #[B, 6]
        }
        return out

    def forward_eval(self, data):
        output = self.forward_model(data)
        best_mode = output['probability'].argmax(dim=-1)
        bs = output['probability'].shape[0]
        output_trajectory = output['trajectory'][torch.arange(bs), best_mode]
        angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
        output["output_trajectory"] = torch.cat(
            [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
        )
        return output
    
    def forward_train(self, data):
        output = self.forward_model(data)
        out = {
            "trajectory": output["trajectory"],
            "probability": output["probability"],
        }


        return out
