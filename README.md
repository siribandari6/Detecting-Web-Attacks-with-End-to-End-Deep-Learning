# Detecting Web Attacks with End-to-End Deep Learning

## Overview
This project aims to detect web attacks using a combination of Django and various machine learning libraries like TensorFlow and Keras.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Detecting-Web-Attacks-with-End-to-End-Deep-Learning
   ```

2. **Create and activate a virtual environment:**
   On Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   On Unix or MacOS:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Apply migrations:**
   ```bash
   python manage.py migrate
   ```

2. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

3. Open your web browser and go to `http://127.0.0.1:8000` to view the project.

## Project Structure
- `Web/`: Contains the main Django project settings.
- `WebApp/`: Contains the Django application with models, views, and URLs.
- `Dataset/`: Example datasets for testing machine learning algorithms.

## Contributing
Feel free to open issues or submit pull requests. Make sure to follow the coding standards used in the project.

## License
This project is licensed under the MIT License.
