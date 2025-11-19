import torch

# This is the pi0.5 policy implementation inside LeRobot
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


MODEL_ID = "lerobot/pi05_base"  # HF repo id for the base π0.5 model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading PI0.5 policy from {MODEL_ID} ...")
    # This pulls config + safetensors weights from Hugging Face Hub
    policy = PI05Policy.from_pretrained(MODEL_ID)

    # Move to your 5090
    policy.to(device)
    policy.eval()

    # Simple sanity checks
    n_params = sum(p.numel() for p in policy.parameters())
    first_param = next(policy.parameters())

    print(f"✅ Loaded PI0.5 with ~{n_params/1e6:.1f}M parameters")
    print(f"First parameter device: {first_param.device}")

    # Optional: print dtype too
    print(f"Parameter dtype: {first_param.dtype}")


if __name__ == "__main__":
    main()