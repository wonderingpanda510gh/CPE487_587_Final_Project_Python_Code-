# Multi-Stage Neural Pipeline for Student Behavioral Clustering and Performance Prediction
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This repository implements a two-stage deep learning architecture designed to analyze student behavior and predict academic outcomes. By combining Unsupervised Deep Clustering with Supervised neural netwrok Regression.

### Project Motivation
Predicting student performance is a classic engineering problem in Educational Informatics, we always want to know whether a student perform good or bad. So I want to use this dataset to first doing cluster, to say what type of student we can have through analyze the dataset, second, I want to predicet the exam grade by using other features, see the relation between other features and exam grade.

### Dataset Description
- Name: Student Performance and Learning Behavior Dataset.
- Samples: 14,003 records (Real-world scale).
- Format: Table Data (.csv).
- Source: *Najem, Kamal. Student Performance and Learning Behavior Dataset for Educational Analytics. Zenodo. https://doi.org/10.5281/zenodo.16459132*

#### Feature Description
| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| **StudyHours** | Number of study hours per week | Numeric |
| **Attendance** | Percentage of classes attended | Numeric |
| **Resources** | Availability and use of academic resources (e.g., library, notes) | Categorical |
| **Extracurricular** | Participation in extracurricular activities | Binary (Yes/No) |
| **Motivation** | Self-reported motivation level | Numeric Scale |
| **Internet** | Access to the internet for study purposes | Binary (Yes/No) |
| **Gender** | Student’s gender | Categorical |
| **Age** | Age of student (18–30 years) | Numeric |
| **LearningStyle** | Preferred learning style (Visual, Auditory, Kinesthetic, etc.) | Categorical |
| **OnlineCourses** | Participation in online courses | Binary (Yes/No) |
| **Discussions** | Engagement in study group discussions or forums | Numeric Scale |
| **AssignmentCompletion** | Rate of completing assignments on time | Numeric Scale |
| **EduTech** | Usage of educational technology tools/platforms | Numeric Scale |
| **StressLevel** | Self-reported stress level | Numeric Scale |
| **ExamScore** | Score obtained in the main exam (Target 1) | Numeric |
| **FinalGrade** | Final course grade (Target 2) | Numeric |