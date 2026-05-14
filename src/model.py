import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Khối Residual"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        
        self.fc_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc_block(x)
        out += residual 
        return self.relu(out)

class SiameseResidualMLP(nn.Module):
    def __init__(self, input_dim=1812, latent_dim=128):
        """
        1812 -> Lớp mở rộng (1536) -> ResBlock(1536) -> ResBlock(768) -> ResBlock(256) -> Output(128)
        """
        super(SiameseResidualMLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Các khối Residual sâu và rộng
        self.res1 = ResidualBlock(1536, 1536, dropout_rate=0.5) 
        
        # Block giảm chiều
        self.res2 = ResidualBlock(1536, 768, dropout_rate=0.5)
        
        # Block giảm chiều tiếp theo
        self.res3 = ResidualBlock(768, 256, dropout_rate=0.4)
        
        # Lớp nén cuối cùng tạo Latent Vector
        self.final_compress = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        latent = self.final_compress(x)
        latent = nn.functional.normalize(latent, p=2, dim=1)
        return latent

if __name__ == "__main__":
    model = SiameseResidualMLP(input_dim=1812, latent_dim=128)
    dummy_input = torch.randn(8, 1812)
    output = model(dummy_input)
    
    print("Kích thước Output: ", output.shape)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TỔNG SỐ THAM SỐ: {total_params:,} parameters") 