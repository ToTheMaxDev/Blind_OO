import subprocess

process = subprocess.Popen("wsl export LD_LIBRARY_PATH=./SFML-3.0.0/lib && ./audio \"BeepSFX.mp3\" 1.0 1.0 0.0 0.0 0.0", shell=False)

try:
    while True:
        print("Hello World")
except KeyboardInterrupt:
    process.terminate()