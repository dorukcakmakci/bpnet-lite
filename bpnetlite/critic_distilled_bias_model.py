import torch
import torch.nn as nn


def compute_receptive_field(stem_kernel, body_dilations, head_kernel):
    """
    For stride-1 conv stacks used here:
      RF = stem_kernel + 2 * sum(body_dilations) + (head_kernel - 1)
    """
    return int(stem_kernel + 2 * sum(int(d) for d in body_dilations) + (head_kernel - 1))


class LocalErrorModel(nn.Module):
    def __init__(self, variant='rf=31bp', n_filters=48, n_outputs=2, input_len=2114, output_len=1000):
        super(LocalErrorModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.trim = (self.input_len - self.output_len) // 2

        if variant == 'rf=15bp':
            # RF = 3 (stem) + [3 layers * (3-1)] + (7-1) head = 15bp
            self.stem_kernel = 3
            self.body_dilations = [1, 1, 1]
            self.head_kernel = 7
        elif variant == 'rf=31bp':
            # Shallow version of deep: use dilation to hit 31bp with fewer layers
            # RF = 3 (stem) + [2*(1+2+2+4)] + (11-1) head = 31bp
            self.stem_kernel = 3
            self.body_dilations = [1, 2, 2, 4]
            self.head_kernel = 11
        elif variant == 'rf=115bp':
            # Stem-fixed compact model with heavy head:
            # RF = 23 (stem) + [2*(1+2+2+4)] + (75-1) head
            #    = 23 + 18 + 74 = 115bp
            self.stem_kernel = 23
            self.body_dilations = [1, 2, 2, 4]
            self.head_kernel = 75
        elif variant == 'rf=1115bp':
            # BPNet-style local stack (no masked stem), plain k=23 stem.
            # RF = 23 (stem) + [2*(2+4+8+16+32+64+128+256)] + (73-1) head
            #    = 23 + 1020 + 72 = 1115bp
            self.stem_kernel = 23
            self.body_dilations = [2, 4, 8, 16, 32, 64, 128, 256]
            self.head_kernel = 73
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.stem = nn.Sequential(
            nn.Conv1d(4, n_filters, kernel_size=self.stem_kernel, padding=self.stem_kernel // 2),
            nn.ReLU()
        )
            
        # Body: Residual blocks with specific dilations
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=3, 
                          padding=d, dilation=d),
                nn.ReLU()
            ) for d in self.body_dilations
        ])
        
        # "Heavy" Final Head: Local assembly / FIR-style smoothing
        # RF contribution: (kernel_size - 1)
        self.head = nn.Conv1d(
            n_filters, 
            n_outputs, 
            kernel_size=self.head_kernel, 
            padding=self.head_kernel // 2
        )
        self.receptive_field = compute_receptive_field(
            self.stem_kernel,
            self.body_dilations,
            self.head_kernel,
        )

    def forward(self, x):
        """
        Input DNA: (B, 4, 2114)
        Returns: (B, n_outputs, 1000)
        """
        # 1. Stem
        x = self.stem(x)
        
        # 2. Residual Body (Shallow stack)
        for layer in self.body:
            x = x + layer(x)
            
        # 3. Heavy Head
        logits = self.head(x)
        
        # 4. Trim to central window
        # Note: No zero-mean constraint per user request.
        return logits[:, :, self.trim:-self.trim]


if __name__ == "__main__":
    variants = ["rf=15bp", "rf=31bp", "rf=115bp", "rf=1115bp"]
    print("Known LocalErrorModel variants:")
    for variant in variants:
        model = LocalErrorModel(
            variant=variant,
            n_filters=48,
            n_outputs=2,
            input_len=2114,
            output_len=1000,
        )
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            y = model(torch.randn(1, 4, 2114))
        print(
            f"  {variant:9s} | stem={model.stem_kernel:>2d} "
            f"| dilations={model.body_dilations} | head={model.head_kernel:>2d} "
            f"| RF={model.receptive_field:>4d} | out_shape={tuple(y.shape)} "
            f"| n_params={n_params}"
        )

    print("\nCan we build RF=115bp with stem_kernel=23 and head_kernel=75?")
    target_rf = 115
    stem_kernel = 23
    head_kernel = 75

    candidates = [
        [1, 2, 2, 4],      # sum=9
        [1, 1, 1, 2, 4],   # sum=9
    ]

    for dilations in candidates:
        required_sum = (target_rf - stem_kernel - (head_kernel - 1)) / 2
        feasible = (sum(dilations) == required_sum)
        rf = compute_receptive_field(stem_kernel, dilations, head_kernel) if feasible else None
        print(
            f"  dilations={dilations} -> head_kernel={head_kernel} "
            f"| sum(d)={sum(dilations)} required_sum(d)={required_sum} "
            f"| feasible={feasible} | RF={rf}"
        )
