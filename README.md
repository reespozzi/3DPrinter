# Cost-Effective Object Analysis for 3D-Printing

**Dependencies required for successful use** 
* install of python-opencv
* install of python NumPy
* install of python time library
* install of python os library
* install of python sys library

**Run Guide with Examples** 

Once inside the root dir of the download, you can install all requirements with the command `pip3 install -r requirements.txt`.
To run the software tool with an expected model and a real image, enter the first of the example commands shown below.
To run the software tool with an expected model, a real image, and the still background image before the print started, enter the third image parameter as the background image, as shown in command 2. Please note that these commands are for a Linux machine. Commands may vary depending on OS used.

Example Commands:
* Command 1: `python3 dimensions.py ./testImages/benchy_model.png ./testImages/benchy_failed.png`.
* Command 2: `python3 dimensions.py ./testImages/benchy_model.png ./testImages/benchy_model_print.png ./testImages/empty.png`.

**Expected Output**

The system will display the segmented versions of both the real and captured images. It will also display the runtime and percentage difference between the two. A classification of error or no error will then be given. The program can be exited by then pressing any key.


