# OpenCV Webapp

## Features

- Real time face detection using device camera
- Measuring heart rate from a recording of one's fingertip (also displays a graph)
- Face detection on uploaded images
- Responsive website using Bootstrap CSS

## Initial setup

1. Clone this repo: `git clone https://github.com/rishi255/opencv-project`

2. Change directory to the repository: `cd opencv-project`

3. Run the installer:
   - For Windows: `./setup.bat`
   - For Linux: `source setup.sh`

The installer will create a virtual environment for the repository and install all the required packages.

## How to run

1. Activate the virtual environment created during setup:
   - `./venv/Scripts/activate` for Windows
   - `source venv1/bin/activate` for Linux
2. Run the command: `flask run` (or `python3 -m flask run` if the previous one does not work)
3. Launch the URL given in the flask output (https://localhost:5000)
