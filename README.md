# An Investment Strategy with RL
This project is on Windows 11 64x. And it's OK for All Windows.

## Installation Steps

<!-- ### Setting Up VNpy Environment

This guide is based on VNpy. If you haven't used it before, you can create a virtual environment first.

```bash
conda create -n vnpy python==3.10.0
```

Activate the virtual environment:
```bash
conda activate vnpy
``` -->

### TA-Lib Installation Guide on Windows

This guide provides instructions on how to install TA-Lib on a Windows system using Visual Studio. You may get an error if you don't do it, this is because the ta-lib (which is neccessary to vnpy) is for 32-bit systems and is not compatible with 64-bit.

#### Prerequisites
- Download and unzip `ta-lib-0.4.0-msvc.zip`. You can find it [here](https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-msvc.zip/download?use_mirror=cfhcable).
- Visual Studio Community with the Visual C++ feature enabled. Ensure this is checked during installation.

#### Steps
1. **Set Up TA-Lib**:
   - Extract the `ta-lib` folder and place it at the root of your C: drive.
![move it to C:](images/ta-lib_installation/MoveToC.png)

2. **Build TA-Lib**:
   - Open the `Native Tools Command Prompt` for Visual Studio.
   ![Open the `Native Tools Command Prompt` ](images/ta-lib_installation/NativeToolsCommandPrompt.png)
   ![Open the `Native Tools Command Prompt` ](images/ta-lib_installation/NativeToolsCommandPromptOpen.png)
   - Navigate to the TA-Lib source directory by entering:
     ```
     cd /d C:\ta-lib\c\make\cdr\win32\msvc
     ```
   - Build the library by running:
     ```
     nmake
     ```
### vnpy Installation Guide on Windows

## Steps
1. **I have Modified `requirements.txt`**:
   - I have added `vnpy_ctastrategy`, `vnpy_ctabacktester`, and `vnpy_sqlite` to `vnpy-3.9.1/requirements.txt`.So this step you can skip.
   - In the original `vnpy-3.9.1/requirements.txt` in [vnpy GitHub repository](https://github.com/vnpy/vnpy), they are missing and you need to add them.

2. **Install vnpy**:
   - Navigate to the `vnpy-3.9.1` directory.
   - Run the installation batch file:
     ```
     install.bat
     ```
   - To start vnpy, use:
     ```
     cd ..
     python run.py
     ```

This setup will launch vnpy, ready for use. Then a window will pop up.
![Successfully run the `run.py` ](images/vnpy_installation/success.png)

For more detailed information, please refer to the official [vnpy GitHub repository](https://github.com/vnpy/vnpy).

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
