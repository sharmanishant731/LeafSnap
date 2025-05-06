# LeafSnap 🌿: Plant Image Identification 

## Description
LeafSnap is a plant image identification app developed by Team 404Found. The app uses deep learning models to identify plant species from images. It provides scientific names, plant information, and external reference links for each identified plant.

### Features:
- Upload an image to predict the plant species.
- Displays the plant's scientific name and additional information.
- Includes an external link for more details on each plant.
- Multiple Swin Transformer models for ensemble prediction.

## Installation

### Prerequisites:
Ensure you have Python 3.8+ installed on your machine.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repository/leafsnap.git
    cd leafsnap
    ```

2. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Pre-trained Models**:
    - You will need to download the following pre-trained Swin Transformer model files and place them in the same directory as your script:
        - `best_swin_fold1.pth`
        - `best_swin_fold2.pth`
        - `best_swin_fold3.pth`
        - `best_swin_fold4.pth`
        - `best_swin_fold5.pth`
    - If you don't have these models, please refer to the instructions in the repository or contact the project maintainers.

4. **Fonts**:
    - The app uses the **Montserrat** font for styling. The font files `Montserrat-Bold.ttf` and `Montserrat-Regular.ttf` are included in the repository. Ensure that these font files are in the same directory as your script.

### Running the Application:
1. **Start the app**:
    ```bash
    python LeafSnap.py
    ```

2. The application window will open, allowing you to upload a plant image and get the prediction.

## Usage Instructions
1. Click on **Upload Image** to select a plant image from your device.
   - Supported image formats: `.jpg`, `.png`.
   - The image will be automatically resized to fit the model's input dimensions (224x224).
2. Click on **Predict** to identify the plant.
3. The prediction will display the plant's name, scientific name, and a link for more information.

## Dataset Used for Training
The dataset used for training the model consists of 97 plant species, specifically focusing on native Indian plant species. Each image is named according to the corresponding plant species label, and the dataset is divided into training, validation, and testing sets. The dataset includes high-quality images in .jpg and .png formats.
The images were collected from various sources and labeled according to the plant species they represent, though no additional annotations such as scientific names or descriptions were provided in the original dataset.

The images were collected from the following sources:
1. [Indian Medicinal Leaves Image Dataset](https://data.mendeley.com/datasets/748f8jkphb/3): A collection of images representing various medicinal plants native to India.
2. [Medicinal Plant Dataset (Augmented)](https://www.kaggle.com/datasets/vishnuoum/medicinal-plant-dataset-augmented?select=data): An augmented version of the dataset containing additional images for better training performance and generalization.

## Supported Plant Classes
The model can identify 97 different plant species including Aloe Vera, Neem, Tulsi, Mango, Hibiscus, and more. For the complete list of classes, check the [`class_labels.txt`](./class_labels.txt) file or refer to `className.py`.

## File Structure
```plaintext
LeafSnap/
│
├── LeafSnap.py                # Main application file
├── requirements.txt           # List of project dependencies
├── className.py               # Contains scientific names, plant info, and links
├── best_swin_fold1.pth        # Pre-trained Swin Transformer model
├── best_swin_fold2.pth        # Another pre-trained model
├── best_swin_fold3.pth        # Another pre-trained model
├── best_swin_fold4.pth        # Another pre-trained model
├── best_swin_fold5.pth        # Another pre-trained model
├── logobg.png                 # Logo image for the app
├── Montserrat-Bold.ttf        # Font for the app (Bold)
├── Montserrat-Regular.ttf     # Font for the app (Regular)
├── .gitattributes             # Git attributes file
└── README.md                  # Project documentation
``` 
---

## 📊 Model Performance (5-Fold Cross-Validation)

| Metric           | Value                        |
|------------------|------------------------------|
| Avg Accuracy     | **97.30%**                   |
| Accuracy 95% CI  | **96.79% – 97.80%**          |
| Avg F1-Score     | **0.9723**                   |
| F1-Score 95% CI  | **0.9668 – 0.9777**          |

### Highlights:
1. Validation accuracy peaked at **98.26%** in Fold 3.
2. Consistently high F1-Scores across folds (≥ **0.9544**).
3. Best models saved automatically when validation improved.

---

## 👨‍💻 Developers

* Rishabh Jain
* Nishant Sharma
* Girish Kumar

---

## 📬 Contact

For questions, suggestions, or bug reports, please contact us:

* rishabhjain61002@gmail.com
* sharmanishant731@gmail.com
* goyalanshul1890@gmail.com

---
