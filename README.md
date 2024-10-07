# PetFinder.my - Pawpularity Contest

## Project Overview
This project aims to predict the **Pawpularity score** of pet images, which ranges from 0 to 100 and reflects the popularity of a pet. We employ a combination of machine learning and deep learning techniques to build a system capable of predicting these scores, leveraging image feature extraction using the **Swin Transformer** and regression models like **XGBoost** through a stacking approach to enhance performance.

## Project Structure
We use the PetFinder.my dataset provided by Kaggle, applying **Swin Transformer** for image feature extraction and combining it with traditional regression models for Pawpularity score prediction.

### File Structure:
- `train.csv`: Contains training data, including image IDs and Pawpularity scores.
- `test.csv`: Contains test data with image IDs, used for generating submission results.
- `train_images/` and `test_images/`: Folders containing the training and test images.
- `swin_model/`: Pre-trained Swin Transformer model for feature extraction.

### Dependencies:
- **Python 3.8**
- **PyTorch**
- **Transformers (Hugging Face)**: For loading the pre-trained Swin Transformer.
- **Scikit-learn**: For model cross-validation and metrics.
- **XGBoost**: For the stacking regression task.

## Model Workflow
1. **Data Preprocessing**:
   - Load `train.csv` and `test.csv`, and generate file paths for each image.
   - Shuffle the training set to ensure randomized distribution of data for model training.
   - Extract image features using Swin Transformer.

2. **K-Fold Cross-Validation**:
   - Apply **StratifiedKFold** to split the training data into 10 folds, ensuring consistent distribution across folds.
   - For each fold, train base models and record the predictions.

3. **Feature Extraction**:
   - Use **Swin Transformer** to extract features for each image. The final hidden state of the Transformer is averaged to produce feature vectors for the images.

4. **Base Model Training**:
   - Train the following base models on the extracted features:
     - **Ridge Regression**
     - **Random Forest Regressor**
     - **Support Vector Regressor (SVR)**
   - Fit each base model on the training set and generate predictions for both validation and test sets.

5. **Stacking Model**:
   - The predictions from the base models are used as meta-features for the second-level model, an **XGBoost Regressor**.
   - XGBoost is trained on these meta-features to produce the final Pawpularity score predictions for the test set.

6. **Model Evaluation**:
   - The performance of the model is evaluated using Root Mean Squared Error (RMSE) for each fold during cross-validation. The mean RMSE across all folds is reported.


