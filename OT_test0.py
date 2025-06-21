# vla_ot_finetune.py

import torch
import numpy as np
import ot
from sklearn.metrics.pairwise import cosine_similarity

# Dummy models (replace with your actual models)
class DummyOpenVLA:
    def encode_image(self, x):
        # Simulate output after MLP projector
        return torch.nn.functional.normalize(torch.randn(x.size(0), 512), dim=1)

class DummySigLIP:
    def encode_image(self, x):
        return torch.nn.functional.normalize(torch.randn(x.size(0), 512), dim=1)

    def encode_text(self, text_list):
        return torch.nn.functional.normalize(torch.randn(len(text_list), 512), dim=1)

def compute_pseudo_actions(new_imgs, new_texts, old_imgs, old_texts, old_actions,
                           openvla_model, siglip_model, alpha=1.0, beta=1.0, gamma=1.0):
    # Extract embeddings
    vla_img_new = openvla_model.encode_image(new_imgs).detach().cpu().numpy()
    vla_img_old = openvla_model.encode_image(old_imgs).detach().cpu().numpy()
    siglip_img_new = siglip_model.encode_image(new_imgs).detach().cpu().numpy()
    siglip_img_old = siglip_model.encode_image(old_imgs).detach().cpu().numpy()
    siglip_text_new = siglip_model.encode_text(new_texts).detach().cpu().numpy()
    siglip_text_old = siglip_model.encode_text(old_texts).detach().cpu().numpy()

    # Compute cost components
    C_img_img = 1 - cosine_similarity(vla_img_new, vla_img_old)
    C_text_text = 1 - cosine_similarity(siglip_text_new, siglip_text_old)
    C_img_text = 1 - cosine_similarity(siglip_img_new, siglip_text_old)

    # Total cost matrix
    M = alpha * C_img_img + beta * C_text_text + gamma * C_img_text

    # Uniform distributions
    a = np.ones((len(new_imgs),)) / len(new_imgs)
    b = np.ones((len(old_imgs),)) / len(old_imgs)

    # Solve OT
    P = ot.emd(a, b, M)
    pseudo_actions = P @ old_actions
    return pseudo_actions

def main():
    torch.manual_seed(42)
    num_new = 10
    num_old = 20

    # Dummy data
    new_imgs = torch.randn(num_new, 3, 224, 224)
    old_imgs = torch.randn(num_old, 3, 224, 224)
    new_texts = ["pick the cup into the box"] * num_new
    old_texts = ["pick the cup into the plate", "pick the can into the box"] * (num_old // 2)
    old_actions = np.random.randn(num_old, 7)  # 7D action vector

    # Models
    openvla_model = DummyOpenVLA()
    siglip_model = DummySigLIP()

    # Compute pseudo-labels
    pseudo_actions = compute_pseudo_actions(
        new_imgs, new_texts, old_imgs, old_texts, old_actions,
        openvla_model, siglip_model
    )

    print("Pseudo actions (for fine-tuning):\n", pseudo_actions)

if __name__ == "__main__":
    main()
