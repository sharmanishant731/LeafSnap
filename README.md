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
