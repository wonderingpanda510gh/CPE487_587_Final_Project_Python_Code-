import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, random_split

from cpe487587finalproject import get_kaggle_data, StudentDataset, VAE, vae_loss_function, FTTransformer, evaluate_clustering, evaluate_regression

def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Beging Training!!")

    print("Loading and Process Data right now")
    raw_df = get_kaggle_data()
    full_dataset = StudentDataset(raw_df)
    train_loader = DataLoader(full_dataset, batch_size=128, shuffle=False) 
    
    num_continuous_orig = len(full_dataset.num_cols)
    latent_dim = 16
    cat_dims_orig = full_dataset.cat_dims


    vae = VAE(cat_dims_orig, num_continuous_orig, latent_dim).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=0.001)
    
    print("*"*20)
    print("Training VAE")
    print("*"*20)
    vae.train()
    for epoch in range(10000):
        for x_cat, x_num, _ in train_loader:
            loss_last = 10e4
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            optimizer_vae.zero_grad()
            recon_x, x_orig, mu, logvar, _ = vae(x_cat, x_num)
            loss = vae_loss_function(recon_x, x_orig, mu, logvar)

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            if (epoch+1) % 50 == 0:
                distance = abs(loss_last - loss.item())  # compute the change in loss from the last iteration
                if distance < 0.0001:  # if the change is smaller than the threshold, stop training
                    break
                else:
                    loss_last = loss.item()  # update the last loss value
            loss.backward()
            optimizer_vae.step()

    vae.eval()
    all_latent = []
    with torch.no_grad():
        for x_cat, x_num, _ in train_loader:
            _, _, mu, _, _ = vae(x_cat.to(device), x_num.to(device))
            all_latent.append(mu.cpu())
    
    latent_space = torch.cat(all_latent, dim=0).numpy()
    
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_space)
    
    evaluate_clustering(latent_space, cluster_labels, output_dir="results")


    cluster_tensor = torch.tensor(cluster_labels, dtype=torch.long).unsqueeze(1)
    latent_tensor = torch.tensor(latent_space, dtype=torch.float32)
    
    new_cat_dims = cat_dims_orig + [num_clusters]
    new_num_cont = num_continuous_orig + latent_dim
    
    enhanced_x_cat = torch.cat([full_dataset.cat_tensor, cluster_tensor], dim=1)
    enhanced_x_num = torch.cat([full_dataset.num_tensor, latent_tensor], dim=1)
    y_target = full_dataset.target_tensor
    
    # create a new dataset and dataloader for the enhanced features
    enhanced_dataset = TensorDataset(enhanced_x_cat, enhanced_x_num, y_target)
    model_ft = FTTransformer(new_cat_dims, new_num_cont, token_dim=32).to(device)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    fft_loss_function = torch.nn.MSELoss()
    total_size = len(enhanced_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(enhanced_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("*"*20)
    print("Training FT-Transformer")
    print("*"*20)
    model_ft.train()
    for epoch in range(10000):
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            optimizer_ft.zero_grad()
            preds = model_ft(batch_cat, batch_num)
            loss = fft_loss_function(preds, batch_y)
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            if (epoch+1) % 50 == 0:
                distance = abs(loss_last - loss.item())  # compute the change in loss from the last iteration
                if distance < 0.0001:  # if the change is smaller than the threshold, stop training
                    break
                else:
                    loss_last = loss.item()  # update the last loss value
            loss.backward()
            optimizer_ft.step()


    model_ft.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_cat, batch_num, batch_y in test_loader:
            p = model_ft(batch_cat.to(device), batch_num.to(device))
            all_preds.append(p.cpu())
            all_trues.append(batch_y)

    evaluate_regression(
        torch.cat(all_trues).numpy(), 
        torch.cat(all_preds).numpy(), 
        target_names=["ExamScore", "FinalGrade"],
        output_dir="results/outcomes"
    )
    print("Pipeline executed successfully. Results saved in 'results/outcomes/'.")

if __name__ == "__main__":
    run_pipeline()