## Setup

```bash
conda create -n blowx python=3.6 -y
conda activate blowx
conda install -c anaconda openblas -y
pip install -r requirements.txt
```

## Usage

```python
python play.py
```
Note: conda environment need to be activate before you execute this command

## Data Collection
Got alalysis folder
Collect Data: analysis/collect.py
Data Extraction:
analysis/Tain.ipynb
analysis/Blow Model.ipynb

### macOS
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask cmake
brew install boost
```
