import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_kaggle_data():
    print("Get dataset via KaggleHub...")
    file_path = "student_performance.csv"
    df_pandas = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "adilshamim8/student-performance-and-learning-style",
        file_path
    )
    df = pl.from_pandas(df_pandas)
    
    # drop null values
    df = df.drop_nulls() 
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

class StudentDataset(Dataset):
    def __init__(self, df):
        df = df
        
        # define the categorical and numerical columns based on the dataset we used
        self.cat_cols = ['Resources', 'Extracurricular', 'Internet', 'Gender', 'LearningStyle', 'OnlineCourses']
        self.num_cols = ['StudyHours', 'Attendance', 'Motivation', 'Age', 'Discussions', 'AssignmentCompletion', 'EduTech', 'StressLevel']
        self.target_cols = ['ExamScore', 'FinalGrade'] # here are the two target columns we want to predict
        
        # standardization for numerical features, we used the function from sklearn
        self.scaler = StandardScaler()
        num_data = self.scaler.fit_transform(df.select(self.num_cols).to_numpy())
        self.num_tensor = torch.tensor(num_data, dtype=torch.float32)
        
        # label encoding for categorical features, we used LabelEncoder from sklearn for simplicity, which will convert each category to an integer.
        self.label_encoders = {}
        cat_data = []
        for col in self.cat_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].to_numpy())
            self.label_encoders[col] = le
            cat_data.append(encoded)
            
        self.cat_tensor = torch.tensor(cat_data, dtype=torch.long).T
        
        # the target tensor, we convert the target columns to a tensor as well, and we use float32 since the targets are continuous values (ExamScore and FinalGrade)
        self.target_tensor = torch.tensor(df.select(self.target_cols).to_numpy(), dtype=torch.float32)
        
        # store the dimension of each categorical feature, which will be used for the embedding layers in the model
        self.cat_dims = [len(le.classes_) for col, le in self.label_encoders.items()]

    def __len__(self):
        return len(self.num_tensor)

    def __getitem__(self, idx):
        return self.cat_tensor[idx], self.num_tensor[idx], self.target_tensor[idx]

def get_dataloader(csv_path, batch_size=64):
    dataset = StudentDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.cat_dims