import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentBilinearAttentionRegularizer(nn.Module): 
    def __init__(self, split_indices, k_models, epsilon=1e-6):
        super().__init__()
        self.split_indices = split_indices
        self.k_models = k_models
        self.epsilon = epsilon  # Prevent numerical instability

        # Define projection layers for each segment
        self.proj_layers = nn.ModuleList()
        for i in range(len(split_indices) - 1):
            d_in = split_indices[i+1] - split_indices[i]
            self.proj_layers.append(nn.Linear(d_in, k_models, bias=False))

        # Initialize learnable attention matrix for bilinear attention
        self.W_att = nn.Parameter(torch.randn(k_models, k_models))  # Bilinear attention matrix

    def forward(self, meta_features, weights):
        """
        Calculate the segment similarity penalty with Bilinear Attention mechanism.
        Only compute the similarity between the meta feature segments and the weight vector.
        """
        B = meta_features.size(0)  # Batch size
        sim_list = []

        # Process each segment of the meta features
        for i, proj in enumerate(self.proj_layers):
            start, end = self.split_indices[i], self.split_indices[i + 1]
            segment = meta_features[:, start:end]  # Extract the segment from meta_features

            # Project the segment into the weight space
            segment_proj = proj(segment)  # [batch_size, k_models]

            # Bilinear attention: apply learnable weight matrix W_att
            # Compute the bilinear similarity between segment_proj and weights
            sim = torch.matmul(segment_proj, self.W_att)  # [batch_size, k_models]
            sim = F.cosine_similarity(sim, weights, dim=-1)  # Cosine similarity across models
            sim = sim ** 2  # Square the similarity to emphasize higher similarities
            sim_list.append(sim)

        # Compute the regularization term by averaging the similarity between the segments and weights
        reg = 0.0
        for sim in sim_list:
            reg += sim.mean()  # Average similarity for each segment

        # Normalize the regularization term by the number of segments
        reg = reg / len(sim_list)  # Average over all segments

        # Ensure numerical stability by clipping the result if it's too large
        reg = torch.clamp(reg, min=self.epsilon)  # Avoid potential overflow
        return reg

def compute_balance_regularization(weight_batch):
    B, K = weight_batch.size()
    Sk = weight_batch.sum(dim=0)
    Sk_ratio = Sk / Sk.sum()
    uniform = torch.full_like(Sk_ratio, 1.0 / K)
    return ((Sk_ratio - uniform) ** 2).mean()

def compute_prediction_diversity(pred_list):
    K = len(pred_list)
    reg = 0.0
    for i in range(K):
        for j in range(i+1, K):
            sim = F.cosine_similarity(pred_list[i], pred_list[j], dim=-1)
            reg += (sim ** 2).mean()
    return reg * 2 / (K * (K - 1))