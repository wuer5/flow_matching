import argparse

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from my_unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Rectified Flow Training Script")

    parser.add_argument("--dataset_path", type=str, default="/home/jianghaoyan/code/datasets/ffhq-64x64",
                        help="Path to dataset directory")
    # 训练参数
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--inference_batch_size", type=int, default=64,
                        help="Inference batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--sigma_min", type=float, default=0,
                        help="Minimum sigma value")
    # 设备参数
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA (use CPU instead)")
    # 随机种子
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    DEVICE = torch.device(f"cuda:{args.gpu_id}" if cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.dataset_path, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    FM = ExactOptimalTransportConditionalFlowMatcher()

    model = UNet(in_ch=3, out_ch=3).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("Start training CFM...")
    model.train()
    best_loss = float('inf')

    for epoch in range(args.n_epochs):
        total_loss = 0
        for batch_idx, (x1, _) in enumerate(tqdm(train_loader, ncols=100)):
            optimizer.zero_grad()
            x1 = x1.to(DEVICE)
            x0 = torch.randn_like(x1).to(DEVICE)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt)
            loss = torch.nn.functional.mse_loss(vt, ut)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        print("\tEpoch", epoch + 1, "complete!", "\tCFM Loss: ", epoch_loss)
        # Saved best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch
            }, 'best_model.pth')
            print(f"New best model saved with loss: {best_loss:.4f}")
    print("Finish!!")


if __name__ == "__main__":
    main()
