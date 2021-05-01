echo "Installing python3-venv package..."
sudo apt install python3-venv > /dev/null
echo "Creating virtual environment..."
python3 -m venv venv1 > /dev/null
source venv1/bin/activate
echo "Ensuring pip is the latest version..."
python3 -m pip install --upgrade pip > /dev/null
echo "Installing requirements for the virtual environment..."
python3 -m pip install -r requirements.txt > /dev/null