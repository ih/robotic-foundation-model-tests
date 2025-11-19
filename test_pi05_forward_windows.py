import torch
from lerobot.policies.pi05 import PI05Policy  # or from lerobot.policies.pi05.modeling_pi05 import PI05Policy

MODEL_ID = "lerobot/pi05_base"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading PI0.5 policy from {MODEL_ID} ...")
    policy = PI05Policy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    print("✅ PI0.5 loaded")

    # ---- Fake SO-101-style observation ----
    B = 1
    H = W = 224

    base_0_rgb = torch.rand(B, 3, H, W, device=device)
    left_wrist_0_rgb = torch.rand(B, 3, H, W, device=device)
    right_wrist_0_rgb = torch.rand(B, 3, H, W, device=device)

    state = torch.zeros(B, 32, device=device)
    task = ["pick up the yellow object and place it on the plate"]

    # Stub language tokens & mask
    lang_len = 32  # arbitrary, just needs to be > 0
    lang_tokens = torch.zeros(B, lang_len, dtype=torch.long, device=device)
    lang_mask = torch.ones(B, lang_len, dtype=torch.bool, device=device)

    batch = {
        "observation.images.base_0_rgb": base_0_rgb,
        "observation.images.left_wrist_0_rgb": left_wrist_0_rgb,
        "observation.images.right_wrist_0_rgb": right_wrist_0_rgb,
        "observation.state": state,
        "task": task,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }

    with torch.inference_mode():
        action = policy.select_action(batch)

    print("✅ select_action completed")
    print(f"Action shape: {action.shape}")
    print(f"Action dtype: {action.dtype}")
    print(f"Action device: {action.device}")

if __name__ == "__main__":
    main()

