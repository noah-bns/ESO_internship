python -m venv env_hci
source env_hci/bin/activate
pip install -r requirements.txt
cd src/pipeline_ADI
pip install -e.


Then in VS Code:
Ctrl+Shift+P
Python: Select Interpreter
Choose /home/aosimul/noah/env_hci/bin/python
