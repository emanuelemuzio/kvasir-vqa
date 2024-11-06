# vqa

python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
deactivate
ollama create kvasirvqa -f ./Modelfile 