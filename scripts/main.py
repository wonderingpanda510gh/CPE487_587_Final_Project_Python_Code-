import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import polars as pl
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

    print("*"*20)
    print("Run Baselines")
    print("*"*20)

    X_baseline = np.concatenate([full_dataset.cat_tensor.numpy(), full_dataset.num_tensor.numpy()], axis=1)
    y_baseline = full_dataset.target_tensor.numpy()

    from sklearn.model_selection import train_test_split
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_baseline, y_baseline, test_size=0.2, random_state=42
    )
    baseline_results = []

    # linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_base, y_train_base)
    lr_preds = lr_model.predict(X_test_base)
    lr_rmse = np.sqrt(mean_squared_error(y_test_base, lr_preds))
    lr_r2 = r2_score(y_test_base, lr_preds)
    print(f"Linear Regression: RMSE: {lr_rmse:.4f} ; R2: {lr_r2:.4f}")
    baseline_results.append({"Model": "Linear Regression", "RMSE": lr_rmse, "R2": lr_r2})

    # random forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_base, y_train_base)
    rf_preds = rf_model.predict(X_test_base)

    rf_rmse = np.sqrt(mean_squared_error(y_test_base, rf_preds))
    rf_r2 = r2_score(y_test_base, rf_preds)
    print(f"Random Forest:  RMSE: {rf_rmse:.4f} ; R2: {rf_r2:.4f}")
    baseline_results.append({"Model": "Random Forest", "RMSE": rf_rmse, "R2": rf_r2})

    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_metrics.csv"

    df_baseline = pl.DataFrame(baseline_results)
    df_baseline.write_csv(output_file)

    print("Baseline models end!!!!!!!!!!!!!!!")


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
    # model_ft = FTTransformer(new_cat_dims, new_num_cont, token_dim=32).to(device)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
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
    results_rmse = []
    results_r2 = []

    for i in range(3):
        print(f"Run {i + 1} of 3")
        model_ft = FTTransformer(new_cat_dims, new_num_cont, token_dim=32).to(device)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
        model_ft.train()
        for epoch in range(10000):
            for batch_cat, batch_num, batch_y in train_loader:
                batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
                optimizer_ft.zero_grad()
                preds = model_ft(batch_cat, batch_num)
                loss = fft_loss_function(preds, batch_y)
                if (epoch+1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

                loss.backward()
                optimizer_ft.step()


        model_ft.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in test_loader:
                p = model_ft(batch_cat.to(device), batch_num.to(device))
                all_preds.append(p.cpu())
                all_trues.append(batch_y)

        all_trues_np = torch.cat(all_trues).numpy()
        all_preds_np = torch.cat(all_preds).numpy()
        
        # each run, we compute the RMSE and R2 for the predictions and print them out
        current_rmse = np.sqrt(mean_squared_error(all_trues_np, all_preds_np))
        current_r2 = r2_score(all_trues_np, all_preds_np)
        print(f"Run {i + 1} Completed: RMSE: {current_rmse:.4f}, R2: {current_r2:.4f}")
        
        results_rmse.append(current_rmse)
        results_r2.append(current_r2)

        if i == 2:
            evaluate_regression(
                torch.cat(all_trues).numpy(), 
                torch.cat(all_preds).numpy(), 
                target_names=["ExamScore", "FinalGrade"],
                output_dir="results"
            )
            print("Pipeline executed successfully. Results saved in 'results'.")

    # compute the 95 confidencen interval for the RMSE and R2
    mean_rmse = np.mean(results_rmse)
    std_rmse = np.std(results_rmse)
    ci_rmse = st.t.interval(0.95, df=len(results_rmse)-1, loc=mean_rmse, scale=st.sem(results_rmse))

    mean_r2 = np.mean(results_r2)
    std_r2 = np.std(results_r2)
    ci_r2 = st.t.interval(0.95, df=len(results_r2)-1, loc=mean_r2, scale=st.sem(results_r2))

    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # record the RMSE and R2 for each run in a CSV file
    runs_data = {
        "Run": [i + 1 for i in range(3)],
        "RMSE": results_rmse,
        "R2": results_r2
    }
    df_runs = pd.DataFrame(runs_data)
    df_runs.to_csv(output_dir / "3_run_metrics.csv", index=False)

    # final 95 confidence interval 
    summary_text = (
        f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} (95% CI: [{ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}])\n"
        f"R2:   {mean_r2:.4f} ± {std_r2:.4f} (95% CI: [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}])\n"
    )

    with open(output_dir / "3_run_statistical_final_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"Pipeline executed successfully. Statistical results saved in: {output_dir}")
if __name__ == "__main__":
    run_pipeline()